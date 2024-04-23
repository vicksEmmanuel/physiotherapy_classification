from dellma.agent.agent import ActionConfig, PreferenceConfig, StateConfig
from dellma.agent.farmagent import FarmAgent
from dellma.agent.physiotherapyagent import GradeAgent
from dellma.agent.tradeagent import TradeAgent
from util.config import CFG
from util.load_dataloader import prepare_ava_dataset
from pytorchvideo.data.clip_sampling import ClipInfo, RandomClipSampler
from pytorchvideo.data.ava import TimeStampClipSampler;

if __name__ == "__main__":
    # # Example to produce the belief distribution prompt
    agent = GradeAgent(
        choices=["good", "brief", "average"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(),
    )
    belief_distribution_prompt = agent.prepare_belief_dist_generation_prompt()

    # Example to produce the full dellma prompt
    agent = GradeAgent(
        choices=["good", "brief", "average"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(pref_enum_mode="order", sample_size=50),
    )
    dellma_prompt = agent.prepare_dellma_prompt()

    with open('testing_output_3.txt', 'w') as file:
        for option in dellma_prompt:
            file.write(option)
    file.close()
