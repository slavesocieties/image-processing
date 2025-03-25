import os
import random
import boto3
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import NoCredentialsError, ClientError

def download_s3_objects_excluding_prefixes(bucket_name, local_dir, exclude_prefixes=None):
    """
    Download all objects from an S3 bucket to a local directory, excluding keys that begin with any of the exclude_prefixes.

    Parameters:
    - bucket_name (str): Name of the S3 bucket
    - local_dir (str): Path to the local directory to download files to
    - exclude_prefixes (list of str, optional): List of prefixes to exclude from download
    """
    s3 = boto3.client('s3')

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    if exclude_prefixes is None:
        exclude_prefixes = []

    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if any(key.startswith(prefix) for prefix in exclude_prefixes):
                    continue

                # Preserve folder structure in key
                local_path = os.path.join(local_dir, key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                print(f"Downloading {key} to {local_path}")
                s3.download_file(bucket_name, key, local_path)

    except NoCredentialsError:
        print("Credentials not available.")
    except ClientError as e:
        print(f"Client error: {e}")

def download_random_s3_objects(bucket_name, local_dir, count):
    """
    Download a specified number of random unique objects from the S3 bucket to the local directory.

    Parameters:
    - bucket_name (str): Name of the S3 bucket
    - local_dir (str): Path to the local directory to download files to
    - count (int): Number of unique objects to download
    """
    s3 = boto3.client('s3')

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    def download_object(key):
        local_path = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading random object {key} to {local_path}")
        s3.download_file(bucket_name, key, local_path)

    try:
        paginator = s3.get_paginator('list_objects_v2')
        reservoir = []
        i = 0

        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                key = obj['Key']
                i += 1
                if len(reservoir) < count:
                    reservoir.append(key)
                else:
                    j = random.randint(0, i - 1)
                    if j < count:
                        reservoir[j] = key

        if not reservoir:
            print("No objects found in the bucket.")
            return

        with ThreadPoolExecutor(max_workers=10) as executor:
            for key in reservoir:
                executor.submit(download_object, key)

    except NoCredentialsError:
        print("Credentials not available.")
    except ClientError as e:
        print(f"Client error: {e}")
