from .base import TaskResult
from .camels_back import camels_back
from .diversity import dont_repeat_yourself
from .free_association import free_association
from .odd_one_out import odd_one_out
from .shaggy_dog import shaggy_dog
from .style_transfer import style_transfer
from .subversion import subversion
from .telephone import telephone_game

TASKS = {
    "free_association": free_association,
    "odd_one_out": odd_one_out,
    "telephone": telephone_game,
    "camels_back": camels_back,
    "diversity": dont_repeat_yourself,
    "style_transfer": style_transfer,
    "subversion": subversion,
    "shaggy_dog": shaggy_dog,
}

__all__ = ["TaskResult", "TASKS", *TASKS.keys()]
