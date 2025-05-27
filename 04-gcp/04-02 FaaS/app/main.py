# [START setup]
import base64
import json
import os

from google.cloud import storage
from google.cloud import vision
import google.cloud.vision_v1

vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# [END setup]

# [START common functions]
def get_image_from_bucket(bucket):
    """Get image from bucket"""
    
    filename = bucket['name']
    image = google.cloud.vision_v1.types.Image()
    image.source.image_uri = f"gs://{bucket['bucket']}/{filename}"

    return filename, image


def save_results(filename, result, extension='.json'):
    """Save results to bucket"""

    bucket_name = config['RESULT_BUCKET']
    result_filename = f'{filename}{extension}'
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(result_filename)

    print(f'Saving results to {result_filename} in bucket {bucket_name}')

    blob.upload_from_string(json.dumps(result))
# [END common functions]

# [START functions_localize_objects]
def localize_objects(request):
    """Localize objects in the image on Google Cloud Storage

    Args:
    request: Cloud Function request object
    """
    
    # Parse the Cloud Storage event
    try:
        envelope = json.loads(request.data.decode('utf-8'))
        
        # Check if it's a Pub/Sub message format
        if 'message' in envelope and 'data' in envelope['message']:
            data = json.loads(base64.b64decode(envelope['message']['data']).decode('utf-8'))
        # Check if it's direct Cloud Storage event format
        elif 'bucket' in envelope and 'name' in envelope:
            data = envelope
        else:
            print(f"Unknown message format: {envelope}")
            return 'Unknown message format', 400
            
    except Exception as e:
        print(f"Error parsing message: {e}")
        return 'Error parsing message', 400
    
    # Extract bucket info
    bucket = {
        'bucket': data['bucket'],
        'name': data['name']
    }
   
    filename, image = get_image_from_bucket(bucket)
    objects = vision_client.object_localization(image=image).localized_object_annotations

    print(f'Number of objects found: {len(objects)}')
    for object_ in objects:
        print(f'\n{object_.name} (confidence: {object_.score})')
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f' - ({vertex.x}, {vertex.y})')

    result = {}
    for i, object_ in enumerate(objects):
        vertices = []
        for vertex in object_.bounding_poly.normalized_vertices:
            vertices.append((vertex.x, vertex.y))
        result[i] = {
          'name': object_.name,
          'score': object_.score,
          'vertices': vertices
        }

    save_results(filename, result, extension='.objects.json')
    return 'OK', 200
# [END functions_localize_objects]


# [START functions_detect_faces]
def detect_faces(request):
    """Detects faces in a file located in Google Cloud Storage

    Args:
    request: Cloud Function request object
    """
    
    # Parse the Cloud Storage event
    try:
        envelope = json.loads(request.data.decode('utf-8'))
        
        # Check if it's a Pub/Sub message format
        if 'message' in envelope and 'data' in envelope['message']:
            data = json.loads(base64.b64decode(envelope['message']['data']).decode('utf-8'))
        # Check if it's direct Cloud Storage event format
        elif 'bucket' in envelope and 'name' in envelope:
            data = envelope
        else:
            print(f"Unknown message format: {envelope}")
            return 'Unknown message format', 400
            
    except Exception as e:
        print(f"Error parsing message: {e}")
        return 'Error parsing message', 400
    
    # Extract bucket info
    bucket = {
        'bucket': data['bucket'],
        'name': data['name']
    }

    filename, image = get_image_from_bucket(bucket)

    response = vision_client.face_detection(image=image)
    faces = response.face_annotations
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')
    for face in faces:
        print(f'anger: {likelihood_name[face.anger_likelihood]}')
        print(f'joy: {likelihood_name[face.joy_likelihood]}')
        print(f'surprise: {likelihood_name[face.surprise_likelihood]}')

        vertices = ([f'({vertex.x},{vertex.y})'
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))

    result = {}
    for i, face_ in enumerate(faces):
        vertices = []
        for vertex in face_.bounding_poly.vertices:
            vertices.append((vertex.x, vertex.y))
        result[i] = {
          'anger': likelihood_name[face_.anger_likelihood],
          'joy': likelihood_name[face_.joy_likelihood],
          'surprise': likelihood_name[face_.surprise_likelihood],
          'vertices': vertices
        }
    
    save_results(filename, result, extension='.faces.json')
    return 'OK', 200
# [END functions_detect_faces]