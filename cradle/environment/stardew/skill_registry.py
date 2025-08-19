import inspect
import base64
from typing import Type, Any

from cradle import constants
from cradle.config import Config
from cradle.environment import SkillRegistry
from cradle.environment import Skill
from cradle.utils.singleton import Singleton
from cradle.log import Logger
import numpy as np

config = Config()
logger = Logger()

SKILLS = {}
def register_skill(name):
    def decorator(skill):
        logger.write(f"Registering skill: {name}")  # 添加这行

        skill_name = name
        skill_function = skill
        skill_code = inspect.getsource(skill)

        # Remove unnecessary annotation in skill library
        if f"@register_skill(\"{name}\")\n" in skill_code:
            skill_code = skill_code.replace(f"@register_skill(\"{name}\")\n", "")

        skill_code_base64 = base64.b64encode(skill_code.encode('utf-8')).decode('utf-8')

        skill_ins = Skill(skill_name,
                          skill_function,
                          np.zeros(2048, dtype=np.float64),  # 使用正确维度的零向量
                          skill_code,
                          skill_code_base64)

        SKILLS[skill_name] = skill_ins

        return skill_ins

    return decorator


class StardewSkillRegistry(SkillRegistry, metaclass=Singleton):

    def __init__(self,
                 *args,
                 skill_configs: dict[str, Any] = config.skill_configs,
                 embedding_provider=None,
                 **kwargs):

        # 在调用父类初始化之前设置 skill_registered
        if skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] is not None:
            skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] = {**SKILLS, **skill_configs[
                constants.SKILL_CONFIG_REGISTERED_SKILLS]}
        else:
            skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] = SKILLS

        super().__init__(skill_configs=skill_configs, embedding_provider=embedding_provider)

        logger.write(f"After super init, hasattr skill_registered: {hasattr(self, 'skill_registered')}")

    def get_all_skill_names(self):
        """返回所有已注册技能的名称列表"""
        return list(self.skills.keys()) if hasattr(self, 'skills') else list(SKILLS.keys())

    def has_skill(self, skill_name):
        """检查技能是否存在"""
        return skill_name in (self.skills if hasattr(self, 'skills') else SKILLS)

    def debug_skill_registry(self):
        """输出技能注册表的调试信息"""
        logger.write(f"SKILLS dict contains: {list(SKILLS.keys())}")
        if hasattr(self, 'skills'):
            logger.write(f"Registry skills contains: {list(self.skills.keys())}")
        else:
            logger.write("Registry has no 'skills' attribute")