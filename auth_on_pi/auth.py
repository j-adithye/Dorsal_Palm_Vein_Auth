"""
auth.py - Registration and verification logic

Registration:
    - Takes 4 captured images (LEFT, CENTRE, LEFT2, CENTRE2)
    - Runs preprocessing + inference on each
    - Computes L2-normalized average embedding
    - Stores 4 individual + 1 average = 5 embeddings total

Verification (1:1):
    - Takes 1 image + username
    - Computes distances to all stored embeddings (avg + 4 individuals)
    - Uses MINIMUM distance — most tolerant of position variation
    - Accepts if min distance < threshold

Identification (1:N):
    - Takes 1 image
    - Checks against all users' average embeddings
    - Returns closest match if within threshold
"""

import numpy as np
from inference import get_embedding
from embeddings import save_embeddings, load_embeddings, load_all_embeddings, user_exists
import config


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def register(username, images):
    if len(images) != 4:
        return {'success': False, 'message': 'Expected 4 images, got ' + str(len(images))}
    if user_exists(username):
        return {'success': False, 'message': 'User "' + username + '" already registered'}

    print('[auth] Registering user: ' + username)
    embeddings = []
    for i, img in enumerate(images):
        try:
            emb = get_embedding(img)
            embeddings.append(emb)
            print('  Image {}/4 embedded  norm={:.4f}'.format(i + 1, float(np.linalg.norm(emb))))
        except Exception as e:
            return {'success': False, 'message': 'Embedding failed on image ' + str(i + 1) + ': ' + str(e)}

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

    try:
        save_embeddings(username, embeddings, avg_embedding)
    except Exception as e:
        return {'success': False, 'message': 'Failed to save embeddings: ' + str(e)}

    print('[auth] Registered "' + username + '" -- 5 embeddings saved (4 individual + 1 average)')
    return {'success': True, 'message': 'User "' + username + '" registered successfully'}


def verify(username, image):
    if not user_exists(username):
        return {'success': False, 'message': 'User "' + username + '" not found',
                'distance': None, 'matched': None}

    print('[auth] Verifying user: ' + username)
    try:
        query_emb = get_embedding(image)
    except Exception as e:
        return {'success': False, 'message': 'Embedding failed: ' + str(e),
                'distance': None, 'matched': None}

    individual_embs, avg_emb = load_embeddings(username)

    # Compute distances to ALL stored embeddings
    all_embs   = [avg_emb] + individual_embs
    all_labels = ['average'] + [f'individual_{i+1}' for i in range(len(individual_embs))]
    distances  = [_distance(query_emb, emb) for emb in all_embs]

    # Use minimum distance — most tolerant of position variation
    min_idx  = int(np.argmin(distances))
    min_dist = distances[min_idx]
    matched  = all_labels[min_idx]

    print('  Distances:')
    for label, dist in zip(all_labels, distances):
        marker = ' <-- best' if label == matched else ''
        print(f'    {label}: {dist:.4f}{marker}')
    print(f'  Min distance: {min_dist:.4f}  threshold: {config.THRESHOLD:.4f}')

    if min_dist < config.THRESHOLD:
        print('  [ACCEPT] matched on ' + matched)
        return {'success': True, 'message': 'Verified as "' + username + '"',
                'distance': min_dist, 'matched': matched}

    print('  [REJECT] no match found')
    return {'success': False, 'message': 'Verification failed — vein pattern does not match',
            'distance': min_dist, 'matched': None}


def identify(image):
    print('[auth] Running identification...')
    try:
        query_emb = get_embedding(image)
    except Exception as e:
        return {'success': False, 'username': None,
                'message': 'Embedding failed: ' + str(e), 'distance': None}

    all_users = load_all_embeddings()
    if not all_users:
        return {'success': False, 'username': None,
                'message': 'No users registered', 'distance': None}

    best_user = None
    best_dist = float('inf')
    for uname, avg_emb in all_users.items():
        dist = _distance(query_emb, avg_emb)
        print(f'  {uname}: {dist:.4f}')
        if dist < best_dist:
            best_dist = dist
            best_user = uname

    print(f'  Best match: {best_user}  distance: {best_dist:.4f}  threshold: {config.THRESHOLD:.4f}')

    if best_dist < config.THRESHOLD:
        print('  [ACCEPT] identified as "' + best_user + '"')
        return {'success': True, 'username': best_user,
                'message': 'Identified as "' + best_user + '"', 'distance': best_dist}

    print('  [REJECT] no match within threshold')
    return {'success': False, 'username': None,
            'message': 'Could not identify — no matching user found', 'distance': best_dist}