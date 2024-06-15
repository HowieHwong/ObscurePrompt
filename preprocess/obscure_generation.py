import argparse
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
from openai import AzureOpenAI
import yaml


# Load configuration
with open('llm_judge_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string):
    if config['api']['type'] == 'azure':
        client = AzureOpenAI(
            api_key=config['api']['key'],
            api_version=config['api']['version'],
            azure_endpoint=config['api']['endpoint']
        )
        chat_completion = client.chat.completions.create(
            model=config['api']['model'],
            messages=[{"role": "user", "content": string}]
        )
    else:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=config['api']['model'],
            messages=[{"role": "user", "content": string}]
        )
        print(completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.readlines()

    with open(output_file, 'a') as f:
        for el in data:
            res = get_res(
                'Please make the following text more obscure, return revised text only. Here is the text: ' + el)
            f.writelines(res + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some text.')
    parser.add_argument('input_file', type=str, help='The path to the input file containing text data')
    parser.add_argument('output_file', type=str, help='The path to the output file to save processed text')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
