from .base import TaskResult
from .camels_back import camels_back
from .copycat import copycat
from .diversity import dont_repeat_yourself
from .free_association import free_association
from .odd_one_out import odd_one_out
from .quilting import quilting
from .same_but_different import same_but_different
from .shaggy_dog import shaggy_dog
from .style_transfer import style_transfer
from .subversion import subversion
from .telephone import telephone_game
from .this_and_that import this_and_that
from .this_and_that_not import this_and_that_not

TASKS = {
    "same_but_different": same_but_different,
    "free_association": free_association,
    "odd_one_out": odd_one_out,
    "telephone": telephone_game,
    "camels_back": camels_back,
    "diversity": dont_repeat_yourself,
    "style_transfer": style_transfer,
    "this_and_that": this_and_that,
    "this_and_that_not": this_and_that_not,
    "copycat": copycat,
    "quilting": quilting,
    "subversion": subversion,
    "shaggy_dog": shaggy_dog,
}

__all__ = ["TaskResult", "TASKS", *TASKS.keys()]
