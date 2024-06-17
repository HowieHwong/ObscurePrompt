# ObscurePrompt


## Introduction

Recently, Large Language Models (LLMs) have garnered significant attention for their exceptional natural language processing capabilities. However, concerns about their trustworthiness remain unresolved, particularly in addressing ''jailbreaking'' attacks on aligned LLMs. Previous research predominantly relies on scenarios with white-box LLMs or specific and fixed prompt templates, which are often impractical and lack broad applicability. 
In this paper, we introduce a straightforward and novel method, named ObscurePrompt, for jailbreaking LLMs, inspired by the observed fragile alignments in Out-of-Distribution (OOD) data. 
Specifically, we first formulate the decision boundary in the jailbreaking process and then explore how obscure text affects LLM's ethical decision boundary. 
ObscurePrompt starts with constructing a base prompt that integrates well-known jailbreaking techniques. 
Powerful LLMs are then utilized to obscure the original prompt through iterative transformations, aiming to bolster the attack's robustness. 
Comprehensive experiments show that our approach substantially improves upon previous methods in terms of attack effectiveness, maintaining efficacy against two prevalent defense mechanisms. 
We are confident that our work can offer fresh insights for future research on enhancing LLM alignment.

![method](https://github.com/HowieHwong/ObscurePrompt/blob/main/image/method.png)


## Usage

To replicate the experiment results, follow these steps:


### Step 1: Configuration File

#### 1. API Configuration

This section defines the settings for connecting to different APIs. The available options are `azure` and `openai`.

```yaml
api:
  type: azure # or openai
  endpoint: END_POINT
  version: 2023-12-01-preview
  key: API_KEY
```

- `type`: Specifies the API provider. Can be either `azure` or `openai`.
- `endpoint`: The endpoint URL for the API.
- `version`: The version of the API to use.
- `key`: The API key for authentication.

#### 2. Device Configuration

This section specifies the device to be used for running the models.

```yaml
device: cuda
```

- `device`: Specifies the device type. The example uses `cuda`, which is for GPU acceleration.

#### 3. API Keys

This section contains the API keys for various services.

```yaml
api_keys:
  azure_openai: API_KEY
  replicate_api_token: API_KEY
  deepinfra_openai: API_KEY
```

- `azure_openai`: API key for Azure OpenAI service.
- `replicate_api_token`: API key for the Replicate service.
- `deepinfra_openai`: API key for DeepInfra OpenAI service.

#### 4. Model Path Mapping

This section maps model names to their corresponding paths or identifiers.

```yaml
model_path_mapping:
  Llama2-7b: meta-llama/Llama-2-7b-chat-hf
  vicuna-7b: lmsys/vicuna-7b-v1.3
  ChatGPT: gpt-3.5-turbo
  GPT-4: gpt-4-turbo
```


### Step2: Execute the following command:

```shell
python run.py
```

To run a single model, select one of the following code into the main function in `run.py`:

```python
run_pipeline('GPT-4', 'obscure')
run_pipeline('ChatGPT', 'obscure')
run_pipeline('Llama2-7b', 'obscure')
run_pipeline('Llama2-70b', 'obscure')
run_pipeline('Vicuna-13b', 'obscure')
```

### Step3: Evaluate your results

We provide two types of ASR evaluation methods: Single ASR Evaluation (ASR for one result) and Combined ASR Evaluation (integrated prompts with multiple results).

#### Configuration

To run the evaluation, define the `model_list` in `config.yaml` under `evaluation_setting`. Here is an example configuration:

```yaml
evaluation_setting:
  model_list:
    - ChatGPT
    - GPT-4
    - Vicuna-7b
    - Llama2-7b
```

If you want to run combined evaluation, set the `combined_num` in `config.yaml`.


Define your result file path in `config.yaml` under `evaluation_setting`. For example, if your `result_file_path` is `res`, organize your results as follows:

```
res/ChatGPT/...
res/Llama2-7b/...
res/Other_model_in_model_list/...
...
```

#### Running Evaluations


To run the single evaluation, use the following command:

```shell
python script.py single
```


To run the combined evaluation, use the following command:

```shell
python script.py combined
```




## Cite ObscurePrompt

```text

```