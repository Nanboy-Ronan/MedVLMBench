from copy import deepcopy
from collections import defaultdict

from wrappers.agent import AgentMetaWrapper


class UCAgentWrapper(AgentMetaWrapper):
    """
    Hierarchical UC-Agent implemented under AgentMetaWrapper.
    Contains Level-1 / Level-2 / Level-3 diagnosis internally.
    """

    def __init__(self, backbone, args):
        super().__init__(backbone=backbone, args=args)

        self.name = f"UCAgent({getattr(backbone, 'name', args.model)})"

        # initialize system prompt (same logic as original Agent)
        self.system_prompt = (
            "Disclaimer: This task is a research-oriented, educational task. "
            "You must base your responses only on observable image features "
            "and logical reasoning principles."
        )

    def _single_expert_level1(self, question, image, temperature=0.7):

        prompt = f"""
        [Core Identity] You are a professional medical imaging expert.
        [Medical Case] {question}

        [Reasoning Requirements]
        1. Describe key visual findings.
        2. Explain medical implications.
        3. Choose the best option.

        [Strict Output Format]
        #Reasoning: <3-5 sentences>
        #Answer: <Single letter>
        """

        return self._chat(prompt, image, temperature)

    # ==========================================================
    # Level 2
    # ==========================================================

    def _level2_diagnosis(self, question, image, level1_reports):

        combined_report = "\n".join(level1_reports)

        prompt = f"""
        You are a senior expert verifying prior consensus.

        [Medical Case]
        {question}

        [Previous Reports]
        {combined_report}

        Check image evidence and confirm or correct.

        [Output Format]
        #Review Reasoning: <3-5 sentences>
        #Answer: <Single letter>
        """

        response = self._chat(prompt, image, temperature=0.5)
        option = self._extract_option(response)

        return option, response

    # ==========================================================
    # Level 3
    # ==========================================================

    def _level3_diagnosis(self, question, image, level1_dict):

        critics = {}

        # each option gets critic
        for option in level1_dict.keys():
            prompt = f"""
            You are a Critical Analyst reviewing hypothesis {option}.

            [Medical Case]
            {question}

            Explain why this hypothesis may be wrong.

            #Flaws: <3-5 sentences>
            """
            critic_response = self._chat(prompt, image, temperature=0.5)
            critics[option] = critic_response

        # leader adjudication
        critic_text = "\n".join([f"Option {k} Critique:\n{v}" for k, v in critics.items()])

        leader_prompt = f"""
        You are the Lead Adjudicator.

        [Medical Case]
        {question}

        [Critiques]
        {critic_text}

        Compare hypotheses and choose final answer.

        #Final Reasoning: <6-8 sentences>
        #Final Answer: <Single letter>
        """

        leader_response = self._chat(leader_prompt, image, temperature=0.1)

        option = self._extract_option(leader_response)

        return option

    def _hierarchy_diagnosis(self, question, image):

        # ---------------------------
        # Level 1
        # ---------------------------
        level1_outputs = []
        level1_dict = defaultdict(list)

        for _ in range(2):
            response = self._single_expert_level1(question, image, temperature=0.7)
            option = self._extract_option(response)

            level1_outputs.append(response)
            level1_dict[option].append(response)

        # unanimous
        if len(level1_dict) == 1:
            level2_option, level2_report = self._level2_diagnosis(question, image, level1_outputs)

            only_option = list(level1_dict.keys())[0]

            if level2_option == only_option:
                return level2_option, "level-2"

        # ---------------------------
        # Level 3 (debate)
        # ---------------------------
        level3_option = self._level3_diagnosis(question, image, level1_dict)

        return level3_option, "level-3"
