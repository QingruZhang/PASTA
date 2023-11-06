import logging
import random
import json 
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import chain, combinations
from typing import Any, Callable, Sequence, cast

from evaluation import data, metrics, models, precompute
from evaluation.utils import experiment_utils
from evaluation.utils.typing import Dataset, Device, StrSequence

import numpy as np
import scipy.stats
import spacy
import torch
import torch.utils.data
from dataclasses_json import DataClassJsonMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm


@dataclass(frozen=True)
class InstructionEvaluationSample:
    """Wrapper around format evaluation sample."""

    id: str
    prompt: str
    generation: str

    predictions: list[str]
    target: str

    logp_predictions: list[float]
    logp_target: float

    fluency: float
    consistency: float

    instruction_evaluation: dict 
    sample_attn_scores: dict 


@dataclass(frozen=True)
class InstructionEvaluationMetrics(DataClassJsonMixin):
    """Wrapper around aggregated error correction metrics."""

    top1_accuracy: float
    topk_accuracy: float
    k: int

    fluency: metrics.Metric
    consistency: metrics.Metric

    instruction_evaluation: dict 


@dataclass(frozen=True)
class BiosBiasInstructionEvaluationResults(DataClassJsonMixin):
    """Wrapper around error correction benchmark."""

    samples: list[InstructionEvaluationSample]
    metrics: InstructionEvaluationMetrics 
    attentions: dict 


def get_all_substring_index(string, substring):
    index = [i for i in range(len(string)) if string.startswith(substring, i)]
    return index


class InstructionEvaluator(object):
    """
    The Evaluator for instruction following tasks: JSON Formatting and Pronouns Changing. 

    Args:
        task (`str`): The name of evaluation task (`json` or `pronchange`). 
        prompt_idx (`int`): The index of the prompte template. See `DEFAULT_PROMPT`. 

    Returns: 
        InstructionEvaluator
    """
    INSTRUCTION_TASK_NAMES = [
        "json", "pronchange", 
    ]
    DEFAULT_PROMPT = {
        "json": [
            '''
            Answer the occupation of {} and generate the answer as json format. Here is an example: \{{'name': , 'occupation': ,\}}. Now generate the answer:
            ''',
            '''
            Generate the occupation of {} as json format: 
            ''',
            '''
            Answer the occupation of {} and generate the answer as json format: 
            ''',
            '''
            **Answer the occupation of {} and generate the answer as json format. Here is an example: \{{'name': , 'occupation': ,\}}. Now generate the answer:**
            ''',
            '''
            "Answer the occupation of {} and generate the answer as json format. Here is an example: \{{'name': , 'occupation': ,\}}. Now generate the answer:"
            ''',
            '''
            Answer the occupation of {} and generate the answer as json format. Here is an example: \{{"name": , "occupation": ,\}}. 
            ''',
        ],
        "pronchange": [
            '''
            For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:
            ''',
            '''
            For the aforementioned descriptions, replace 'she' and 'he' with 'they' in the aformentioned text and generate the new text after replacing the pronouns:
            ''',
            '''
            Change 'she' and 'he' with 'they' and answer the occupation of {} after replacing the pronouns: 
            ''',
            '''
            **For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:**
            ''',
            '''
            "For the aforementioned text, substitute 'she' and 'he' with 'they' and generate the occupation of {} after changing pronouns:"
            ''',
        ], 
    }
    JSON_ALIKE_CHARS = ["{", "}", ":", ","]
    PRON_CHANGE_TARGET = ["she", "he"] 
    PRON_ALL_CHANGE_TARGET = ["she", "he", "her", "him", "hers", "his", "herself", "himself"] 

    def __init__(
        self, 
        task:str,
        prompt_idx: int|list|None = None, 
    ):
        self.task = task
        assert task in self.INSTRUCTION_TASK_NAMES 

        if isinstance(prompt_idx, int): 
            self.prompt_temple = [self.DEFAULT_PROMPT[task][prompt_idx]]
            self.prompt_idx = [prompt_idx]
        elif isinstance(prompt_idx, list): 
            self.prompt_temple = [self.DEFAULT_PROMPT[task][i] for i in prompt_idx]
            self.prompt_idx = prompt_idx 
        else:
            self.prompt_temple = self.DEFAULT_PROMPT[task]
            self.prompt_idx = list(range(len(self.DEFAULT_PROMPT[task])))
        
        self.prompt_temple = [prompt.strip() for prompt in self.prompt_temple] 
        self.num_prompt = len(self.prompt_temple) 
        self.nlp = spacy.load("en_core_web_sm")
        
    def parapare_prompt_inputs(
            self, 
            contexts: list, 
            entities: list, 
            other_text_inputs: tuple[list], 
        ):
        """Apply the prompt templates and generation input strings for models."""
        prompts = [
            self.apply_prompt_templete(context, entity, prompt, idx)
                for context,entity in zip(contexts, entities) for prompt,idx in zip(self.prompt_temple, self.prompt_idx)
        ]
        instructions = [
            prompt.format(entity) for entity in entities for prompt in self.prompt_temple 
        ]
        new_entities = [
            entity for entity in entities for i in range(self.num_prompt)
        ]
        new_other_inputs = []
        for text_input in other_text_inputs:
            new_text_input = [
                val for val in text_input for i in range(len(self.prompt_temple))
            ]
            new_other_inputs.append(new_text_input)
        
        return prompts, instructions, new_entities, tuple(new_other_inputs)
    
    def evaluate_sample(self, generation, target=None, **kwargs):
        result = {}
        if self.task == "json":
            result["json_EM"] = self.evaluate_json_EM(generation)
            if target is not None:
                result['json_pred_acc'] = self.evaluate_json_pred(generation, target)
            for requrie_num in range(2, len(self.JSON_ALIKE_CHARS)+1):
                result[f"json_alike_{requrie_num}"] = self.evaluate_json_alike(generation, requrie_num)
            return result 
        elif self.task == "pronchange":
            result["pron_sub_acc"], result["pron_all_acc"] = self.evaluate_pronounce_change(generation)
        return result 
    
    def apply_prompt_templete(self, context, entity, prompt_templete, prompt_idx):
        """Apply the prompt template"""
        prompt = context+'\n'+prompt_templete.format(entity)+'\n' 
        if self.task == "json" and prompt_idx in [5]:
            prompt = prompt_templete.format(entity)+'\n'+context+'\n'+"Now generate the answer:"
        else:
            prompt = context+'\n'+prompt_templete.format(entity)+'\n' 
        return prompt
    
    def evaluate_json_alike(self, generation, require_num):
        """
        Evaluate how generations are alike JSON format
        by checking whether generations include '{', ':', ',', '}'. 
        """
        all_combs = list(combinations(self.JSON_ALIKE_CHARS, require_num))
        for chars in all_combs:
            if all([char in generation for char in chars]):
                return True 
        return False 

    def evaluate_json_EM(self, generation):
        """
        Evaluate whether generations exactly match JSON format 
        by `json.loads` function. 
        """
        generation = generation.strip()
        if "{" not in generation and "}" not in generation:
            return False
        
        string_begs = get_all_substring_index(generation, "{")
        string_ends = get_all_substring_index(generation, "}")
        # Try to load every substring starting from '{' and end with '}'
        for idx_beg in string_begs:
            for idx_end in string_ends:
                substring = generation[idx_beg:idx_end+1]
                try:
                    json.loads(substring)
                    return True 
                except ValueError:
                    continue
        return False 
    
    def evaluate_json_pred(self, generation, target):
        """
        Evaluate whether generated JSON outputs correctly include 
        the traget label by checking its `values` after `json.loads`
        """
        generation = generation.strip()
        if "{" not in generation and "}" not in generation:
            return False
        
        string_begs = get_all_substring_index(generation, "{")
        string_ends = get_all_substring_index(generation, "}")
        for idx_beg in string_begs:
            for idx_end in string_ends:
                substring = generation[idx_beg:idx_end+1]
                try:
                    pred = json.loads(substring)
                    pred_values = [p.lower() if isinstance(p, str) else p for p in pred.values()]
                    if target.lower() in pred_values:
                        return True 
                    for value in pred_values:
                        if isinstance(value, str) and target.lower() in value:
                            return True
                except ValueError:
                    continue
        return False 
    
    def evaluate_pronounce_change(self, generation):
        """
        Evaluate the accuracy of successfully changing the pronouns 
        """
        generation = generation.lower()
        all_tokens = [token.text for token in self.nlp(generation)]
        pron_sub, pron_all = True, True
        for word in all_tokens:
            if word in self.PRON_CHANGE_TARGET:
                pron_sub = False
            if word in self.PRON_ALL_CHANGE_TARGET:
                pron_all = False
        return pron_sub, pron_all
        
    def aggregate_evaluation_results(self, samples):
        """Aggergate the evaluation results"""
        result_key = list(samples[0].instruction_evaluation.keys())
        grouped_results = {}
        for sample in samples:
            if sample.id not in grouped_results:
                grouped_results[sample.id] = {key:[] for key in result_key}
            for key in sample.instruction_evaluation:
                grouped_results[sample.id][key].append(sample.instruction_evaluation[key])
        
        all_result = {key:[] for key in result_key}
        for _, sample_eval in grouped_results.items():
            for key in sample_eval:
                all_result[key].append(any(sample_eval[key])) 

        return {key:sum(value)/len(value) for key,value in all_result.items()}
    
    def is_pred_true_sample_result(self, sample):
        if self.task == "json":
            return sample.instruction_evaluation["json_EM"]
        elif self.task == "pronchange":
            return sample.instruction_evaluation["pron_sub_acc"]

    def get_generation_label(self, context, entity, target):
        if self.task == "json":
            label = '{"name":"%s", "occupation":"%s"}'%(entity, target)
        elif self.task == "pronchange":
            label = context.lower().replace('she', 'they').replace(
                'he', 'they').replace('him', 'them').replace('her', 'them')
        return label 

    def prepare_fewshot_examples(
            self, fewshot_samples, example_sep = "\n###\n", text_target_sep = " ", 
        ):
        """Prepare few-shot demonstration"""
        labeled_example = [] 
        for example in fewshot_samples:
            context = example['context']
            entity = example['entity']
            target = example['target_mediated'] 

            label = self.get_generation_label(context, entity, target) 
            labeled_example.append(context+text_target_sep+label)
        
        fewshot_examples = example_sep.join(labeled_example) + example_sep
        return fewshot_examples 


# For attention score recodring 
def concatenate_attention_scores(output_attentions):
    num_layer = len(output_attentions[0]) 
    num_new_tokens = len(output_attentions) 
    all_attn_scores = [] 
    for layer_id in range(num_layer):
        token_len = output_attentions[0][layer_id].size(-1)
        layer_attn_score = [output_attentions[0][layer_id][:,:,-1,:].unsqueeze(2)]
        for tid in range(1, num_new_tokens):
            layer_attn_score.append(
                output_attentions[tid][layer_id][:,:,0,:token_len].unsqueeze(2)
            )
        all_attn_scores.append(torch.cat(layer_attn_score, dim=2))
    return all_attn_scores 


def calcluate_attn_metric(sub_attn_score, metric):
    if metric == "token_mean_head_all":
        return sub_attn_score.mean(-1).tolist()
    elif metric == "token_head_mean":
        return sub_attn_score.mean().tolist()
    elif metric == "token_head_median":
        return sub_attn_score.float().median(-1).values.median().item()
    elif metric == "token_mean_head_median":
        return sub_attn_score.float().mean(-1).median().item()
    elif metric == "token_mean_head_std":
        return sub_attn_score.mean(-1).std().item() 
    else:
        raise ValueError(f"Unimplement the metric {metric}")


def process_sample_attention_score(
        all_attn_scores, tr_dict, idx, 
        sections=["first", "all"],
        metrics=["token_head_median"]
    ):
    sample_attn_scores = {sec: {key:{metric:[] for metric in metrics} for key in tr_dict} for sec in sections}
    for layer_id in range(len(all_attn_scores)):
        for section in sections:
            if section == "first":
                attn_score = all_attn_scores[layer_id][idx,:,0,:]
            else:
                attn_score = all_attn_scores[layer_id][idx,:,:,:].mean(1)
            for key in sample_attn_scores[section]:
                for metric in metrics:
                    assert tr_dict[key][0]<tr_dict[key][1]
                    sample_attn_scores[section][key][metric].append(
                        calcluate_attn_metric(attn_score[:,tr_dict[key][0]:tr_dict[key][1]], metric)
                    )
                sample_attn_scores[section][key]["token_range"] = tr_dict[key] 
    return sample_attn_scores


def average_sample_attention_scores(samples):
    attn_scores = {}
    for sample in samples: 
        sample_attn_scores = sample.sample_attn_scores
        for sec in sample_attn_scores:
            if sec not in attn_scores:
                attn_scores[sec] = {}
            for key in sample_attn_scores[sec]:
                if key not in attn_scores[sec]:
                    attn_scores[sec][key] = {}
                for metric in sample_attn_scores[sec][key]:
                    if metric == "token_range":
                        continue
                    if metric not in attn_scores[sec][key]:
                        attn_scores[sec][key][metric] = []
                    attn_scores[sec][key][metric].append(
                        sample_attn_scores[sec][key][metric]
                    )
    for sec in attn_scores:
        for key in attn_scores[sec]:
            for metric in attn_scores[sec][key]:
                attn_scores[sec][key][metric] = torch.tensor(
                    attn_scores[sec][key][metric]
                ).mean(0).tolist()
    return attn_scores 


def average_sample_attention_scores_by_pred(samples, eval_sample_func):
    all_sample_attentions = average_sample_attention_scores(samples) 
    samples_pred_true = [
        sample for sample in samples if eval_sample_func(sample) 
    ]
    samples_pred_false = [
        sample for sample in samples if not eval_sample_func(sample)
    ]
    true_sample_attentions = average_sample_attention_scores(samples_pred_true)
    false_sample_attentions = average_sample_attention_scores(samples_pred_false)

    for section in list(true_sample_attentions.keys()):
        new_section = f"{section}_by_pred"
        true_sample_attentions[new_section] = true_sample_attentions.pop(section)
        for key in true_sample_attentions[new_section]:
            new_key = f"{key}_true"
            true_sample_attentions[new_section][new_key] = true_sample_attentions[new_section].pop(key)

    for section in list(false_sample_attentions.keys()):
        new_section = f"{section}_by_pred"
        false_sample_attentions[new_section] = false_sample_attentions.pop(section)
        for key in false_sample_attentions[new_section]:
            new_key = f"{key}_false"
            false_sample_attentions[new_section][new_key] = false_sample_attentions[new_section].pop(key)
    
    pred_sample_attentions = {}
    for new_section in true_sample_attentions:
        pred_sample_attentions[new_section] = true_sample_attentions[new_section] | false_sample_attentions[new_section]

    return all_sample_attentions | pred_sample_attentions

def prepare_biasbios_few_shot_examples(
        fewshot_samples, text_key="prompt" , example_sep = "\n###\n", text_target_sep = " "
    ):
    labeled_examples = [] 
    for example in fewshot_samples:
        text = example[text_key]
        entity = example['entity']
        target = example['target_mediated'] 
        text_label = text + text_target_sep + target 
        labeled_examples.append(text_label) 
    
    fewshot_examples = example_sep.join(labeled_examples) + example_sep
    return fewshot_examples 

def prepare_counterfact_fewshot_examples(
        fewshot_samples, 
        add_unmediated_fact="left", unmediated_prefix="Previously ", mediated_prefix="Currently ",
        example_sep = "\n###\n", text_target_sep = " "
    ):
    labeled_examples = [] 
    for example in fewshot_samples:
        prompt = example['prompt']
        context = example['context']
        target_mediated = example['target_mediated']
        target_unmediated = example['target_unmediated']

        if add_unmediated_fact == "left":
            unmediated_fact = context.replace(target_mediated, target_unmediated)+". "
            new_prompt = f"{unmediated_prefix}{unmediated_fact}{mediated_prefix}{prompt}"
        elif add_unmediated_fact == "right":
            prompt = prompt.replace(target_mediated, target_unmediated)
            context = context.rstrip(". ") + ". "
            new_prompt = f"{mediated_prefix}{context}{unmediated_prefix}{prompt}"
        else:
            raise ValueError("Unexpected %s"%(add_unmediated_fact))
        
        text_label = new_prompt + text_target_sep + target_mediated
        labeled_examples.append(text_label) 
    
    fewshot_examples = example_sep.join(labeled_examples) + example_sep
    return fewshot_examples 


        






    
    
        


            

        











