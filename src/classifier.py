import json
import time
from jinja2 import Template
from loguru import logger


class DialogueCriteriaClassifier:
    def __init__(
        self,
        llm_client,
        criteria_dict: dict[str, str],
        prompt_file: str,
        model: str,
        temperature: float = 0.01,
    ):
        self.criteria_dict = criteria_dict
        self.model = model
        self.temperature = temperature
        self.template = self._init_template(prompt_file)

        self.llm_client = llm_client

    def _init_template(self, prompt_file: str) -> Template:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template_str = f.read()
        
        return Template(template_str)

    def _construct_few_shot_prompt(self, desc_per_prompt: int = 300, domain_info: str | None = None) -> str:
        descriptions = [(criteria, description) for criteria, description in self.criteria_dict.items()]
        prompts = []
        for i in range(0, len(descriptions) + desc_per_prompt - 1, desc_per_prompt):
            descriptions_dict = {
                criteria: description for (criteria, description) in descriptions[i: i + desc_per_prompt]
            }
            if len(descriptions_dict) > 0:
                prompts.append(self.template.render(existing_small_tags=descriptions_dict, domain_info=domain_info))
        return prompts

    def _build_json_schema(self) -> dict:
        properties = {
            "thoughts": {"type": "string"},
            "found_criteria": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(self.criteria_dict.keys()),
                },
            },
        }

        schema = {
            "type": "object",
            "properties": properties,
            "required": ["thoughts", "found_criteria"],
            "additionalProperties": False,
        }

        return schema

    def classify_text(self, dialogue: str, structure: str = "llm", domain_info: str | None = None) -> dict:
        if structure == "llm":
            return self.classify_text_llm(dialogue, domain_info=domain_info)
        elif structure == "keywords":
            return self.classify_text_keywords(dialogue)
        return self.classify_text_llm(dialogue, domain_info=domain_info)


    def classify_text_keywords(self, dialogue: str) -> dict:
        result = {}
        for criteria, description in self.criteria_dict.items():
            try:
                keywords = [k.lower() for k in description.split("+")]
                result[criteria] = all([k in dialogue.lower() for k in keywords])
            except:
                logger.error(f"Error parsing rule: {description}")
                result[criteria] = False
        return result


    def classify_text_llm(self, dialogue: str, domain_info: str | None = None) -> dict:
        json_schema = self._build_json_schema()
        system_messages = self._construct_few_shot_prompt(domain_info=domain_info)
        found_all = []
        for system_message in system_messages:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": dialogue}
            ]

            try:
                result = self.llm_client.generate(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    schema=json_schema,
                )
            except Exception as e:
                logger.error(f"Error sending LLM request: {repr(e)}")
                time.sleep(1)
                try:
                    result = self.llm_client.generate(
                        messages,
                        model=self.model,
                        temperature=self.temperature,
                        schema=json_schema,
                    )
                except Exception as e:
                    logger.error(f"Error AGAIN sending LLM request: {repr(e)}")
                    return {label: None for label in self.criteria_dict}

            try:
                parsed_result = json.loads(result) if isinstance(result, str) else result
                found = list(set(parsed_result.get("found_criteria", [])))
                found_all.extend(found)
            except Exception as e:
                logger.error(
                    f"Error parsing LLM response: {repr(e)}. Response: {result}"
                )
        if len(found_all) > 0:
            labels = {label: label in found_all for label in self.criteria_dict}
        else:
            labels = {label: None for label in self.criteria_dict}
        return labels
