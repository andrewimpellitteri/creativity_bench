from base import CreativityBenchmarkBase
from free_association import FreeAssociationMixin
from telephone_game import TelephoneGameMixin
from camels_back import CamelBackMixin
from diversity import DiversityMixin
from style_transfer import StyleTransferMixin
from combined_score import CombinedScoreMixin

class CreativityBenchmark(
    CreativityBenchmarkBase,
    FreeAssociationMixin,
    TelephoneGameMixin,
    CamelBackMixin,
    DiversityMixin,
    StyleTransferMixin,
    CombinedScoreMixin
):
    pass