from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse

from .services import media_form_depends_execute, MediaFormService
from .utils import save_file, process_image, format_data
from .schemas import CarAttributes

media_router = APIRouter(prefix="/api/v1/vision", tags=["AutoVision"])


@media_router.post("/detect/")
async def detect_car_attrs(
    service: Annotated[MediaFormService, Depends(media_form_depends_execute)],
    file: UploadFile = File(...),
):
    """
    Detects car attributes from an uploaded image.

    Args:
        service (MediaFormService): The service to handle media form operations.
        file (UploadFile): The image file to be processed.

    Returns:
        CarAttributes: The detected car attributes as a response.

    Raises:
        HTTPException: If the uploaded file is not an image or if any server error occurs.
    """
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(
    #         status_code=400, detail=f"File '{file.filename}' is not an image."
    #     )

    url, filepath = await save_file(file)
    result = await process_image(filepath)
    try:
        if isinstance(result, dict):
            return JSONResponse({"data": result})
        else:
            return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=200, detail=str(e))
