from typing import TypedDict, Optional
from typing_extensions import Annotated
import operator


def _last(a, b):
    return b


class FoodInvestigationState(TypedDict):
    batch_id: Annotated[str, _last]
    vision_id: Annotated[str, _last]
    aroma_id: Annotated[str, _last]
    taste_id: Annotated[str, _last]
    correlation_id: Annotated[str, _last]
    investigation_id: Annotated[str, _last]
    passport_id: Annotated[str, _last]
    certificate_id: Annotated[str, _last]

    food_type: Annotated[str, _last]
    batch_name: Annotated[str, _last]
    image_path: Annotated[Optional[str], _last]
    aroma_data: Annotated[Optional[dict], _last]
    taste_data: Annotated[Optional[dict], _last]

    vision_result: Annotated[Optional[dict], _last]
    aroma_result: Annotated[Optional[dict], _last]
    taste_result: Annotated[Optional[dict], _last]
    correlation_result: Annotated[Optional[dict], _last]
    investigation_result: Annotated[Optional[dict], _last]
    review_result: Annotated[Optional[dict], _last]
    passport_result: Annotated[Optional[dict], _last]

    errors: Annotated[list, operator.add]
