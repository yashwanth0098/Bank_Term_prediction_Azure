import os
from azure.storage.blob import BlobServiceClient


class AzureBlobSync:
    """
    Azure Web App compatible replacement for S3Sync
    Uses env vars injected by GitHub Actions / App Service
    """

    def __init__(self):
        try:
            self.container_name = os.environ["AZURE_CONTAINER_NAME"]
            self.connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        except KeyError as e:
            raise EnvironmentError(f"Missing environment variable: {e}")

        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )

        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        # Ensure container exists
        try:
            self.container_client.create_container()
        except Exception:
            pass  # container already exists

    def sync_folder_to_blob(self, local_folder: str, blob_prefix: str) -> None:
        """
        Equivalent to:
        aws s3 sync local_folder s3://bucket/blob_prefix
        """
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)

                blob_path = os.path.join(
                    blob_prefix,
                    os.path.relpath(local_path, local_folder)
                ).replace("\\", "/")

                with open(local_path, "rb") as data:
                    self.container_client.upload_blob(
                        name=blob_path,
                        data=data,
                        overwrite=True
                    )

    def sync_folder_from_blob(self, local_folder: str, blob_prefix: str) -> None:
        """
        Equivalent to:
        aws s3 sync s3://bucket/blob_prefix local_folder
        """
        os.makedirs(local_folder, exist_ok=True)

        blobs = self.container_client.list_blobs(
            name_starts_with=blob_prefix
        )

        for blob in blobs:
            relative_path = os.path.relpath(blob.name, blob_prefix)
            local_path = os.path.join(local_folder, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as file:
                file.write(
                    self.container_client
                    .download_blob(blob.name)
                    .readall()
                )
