"""
app.py - Flask web application for Dorsal Palm Vein Authentication
Run: python3 app.py
Access: http://palmauth.local:5000
"""
import config
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify

from camera import get_camera, capture, close
from auth import register, verify, identify
from embeddings import list_users, delete_user, user_exists

app = Flask(__name__)

# ── Camera lock — prevent concurrent access ──
_cam_lock = threading.Lock()

# ── In-memory capture store for step-by-step registration ──
_capture_store = {}

# ── Registration positions ──
REG_POSITIONS = ['LEFT', 'CENTRE', 'LEFT 2', 'CENTRE 2']
N_CAPTURES    = len(REG_POSITIONS)


# ────────────────────────────────
#  QUALITY CHECK
# ────────────────────────────────

def check_capture_quality(gray, min_variance=200):
    """
    Returns (is_good, variance).
    High variance = clear vein structure visible.
    """
    try:
        blurred  = cv2.GaussianBlur(gray.astype(np.float32), (11, 11), 0)
        variance = float(np.var(blurred))
        return variance >= min_variance, round(variance, 1)
    except Exception:
        return True, 0.0


# ────────────────────────────────
#  MJPEG STREAM
# ────────────────────────────────

def generate_frames():
    cam = get_camera()
    while True:
        with _cam_lock:
            frame = cam.capture_array("main")
        gray = frame        # Rotate 270 degrees
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        _, jpeg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ────────────────────────────────
#  PAGES
# ────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('register.html', n_captures=N_CAPTURES, positions=REG_POSITIONS)


@app.route('/verify')
def verify_page():
    return render_template('verify.html')


@app.route('/identify')
def identify_page():
    return render_template('identify.html')


@app.route('/admin')
def admin_page():
    users = list_users()
    return render_template('admin.html', users=users)

@app.route('/debug')
def debug_page():
    return render_template('debug.html')

@app.route('/stream_preprocessed')
def stream_preprocessed():
    def generate():
        from inference import _preprocess
        cam = get_camera()
        while True:
            with _cam_lock:
                frame = cam.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                processed = _preprocess(gray)
                # float32 normalized → uint8
                lo, hi = processed.min(), processed.max()
                display = ((processed - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)
                display = cv2.rotate(display, cv2.ROTATE_90_CLOCKWISE)
                display = cv2.rotate(display, cv2.ROTATE_90_CLOCKWISE)
                display = cv2.rotate(display, cv2.ROTATE_90_CLOCKWISE)
                bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f'[stream_preprocessed] error: {e}')
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + jpeg.tobytes() + b'\r\n')
            time.sleep(0.5)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# ────────────────────────────────
#  API ENDPOINTS
# ────────────────────────────────

@app.route('/api/register/capture', methods=['POST'])
def api_register_capture():
    """
    Step-by-step registration — 4 captures (LEFT, CENTRE, LEFT2, CENTRE2).
    Client calls this once per step. On final step, registers the user.
    """
    try:
        data     = request.get_json()
        username = data.get('username', '').strip()
        step     = int(data.get('step', 0))

        if not username:
            return jsonify({'success': False, 'message': 'Username cannot be empty'})

        with _cam_lock:
            img = capture()

        # Quality check
        is_good, variance = check_capture_quality(img)
        if not is_good:
            return jsonify({
                'success':  False,
                'message':  f'Capture quality too low (score={variance}) — reposition hand and try again',
                'retry':    True
            })

        _capture_store[f'{username}_{step}'] = img
        print(f'[register] {username} step {step+1}/{N_CAPTURES}  variance={variance}')

        if step < N_CAPTURES - 1:
            return jsonify({
                'success':       True,
                'message':       f'Step {step+1} captured',
                'next':          True,
                'next_step':     step + 1,
                'next_position': REG_POSITIONS[step + 1]
            })
        else:
            # All captures done
            images = []
            for i in range(N_CAPTURES):
                img_i = _capture_store.pop(f'{username}_{i}', None)
                if img_i is None:
                    for j in range(N_CAPTURES):
                        _capture_store.pop(f'{username}_{j}', None)
                    return jsonify({'success': False, 'message': f'Missing capture {i+1}, please restart'})
                images.append(img_i)

            result = register(username, images)
            return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/verify', methods=['POST'])
def api_verify():
    """1:1 verification."""
    try:
        data     = request.get_json()
        username = data.get('username', '').strip()

        if not username:
            return jsonify({'success': False, 'message': 'Username cannot be empty'})

        if not user_exists(username):
            return jsonify({'success': False, 'message': f'User "{username}" not registered'})

        with _cam_lock:
            img = capture()

        result = verify(username, img)

        dist = result.get('distance')
        if dist is not None:
            from config import THRESHOLD
            confidence =  max(0, min(100, int((1 - dist / config.THRESHOLD) * 100)))
            result['confidence'] = confidence

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/identify', methods=['POST'])
def api_identify():
    """1:N identification."""
    try:
        with _cam_lock:
            img = capture()

        result = identify(img)

        dist = result.get('distance')
        if dist is not None:
            from config import THRESHOLD
            confidence = max(0, min(100, int((1 - dist / THRESHOLD) * 100)))
            result['confidence'] = confidence

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/users', methods=['GET'])
def api_users():
    return jsonify({'users': list_users()})


@app.route('/api/delete', methods=['POST'])
def api_delete():
    try:
        data     = request.get_json()
        username = data.get('username', '').strip()
        if not username:
            return jsonify({'success': False, 'message': 'Username required'})
        result = delete_user(username)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
        
@app.route('/stream_debug')
def stream_debug():
    cam = get_camera()
    with _cam_lock:
        frame = cam.capture_array("main")
    return jsonify({'shape': list(frame.shape)})
    
@app.route('/api/security', methods=['POST'])
def api_security():
    import config
    data = request.get_json()
    high = bool(data.get('high', False))
    config.set_high_security(high)
    mode = 'HIGH' if high else 'NORMAL'
    print(f'[security] Mode set to {mode}')
    return jsonify({'success': True, 'mode': mode})

@app.route('/api/security', methods=['GET'])
def api_security_get():
    import config
    high = config.THRESHOLD == config.THRESHOLD_HIGH
    return jsonify({'high': high, 'mode': 'HIGH' if high else 'NORMAL'})

# ────────────────────────────────
#  MAIN
# ────────────────────────────────

if __name__ == '__main__':
    print('Initializing camera...')
    get_camera()
    print('Camera ready.')
    print(f'Registration: {N_CAPTURES} captures per user — {REG_POSITIONS}')
    print('Starting server at http://palmauth.local:5000')
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        close()