import torch

from dataloader import load, process_input
from transformers import AutoModel

torch.cuda.empty_cache()
bert_transformer = AutoModel.from_pretrained('bert-base-uncased')


def train():
    # preprocess data
    data = load()
    data = data['train'][:100]
    questions = data['question']
    questions, question_seg = process_input(questions, segment=0)

    contexts, contexts_seg = [], []
    for context in data['context_long']:
        content, content_seg = [], []
        # concatenate each document into a single content array and separate them by a separator token and segment id
        content = "".join(context['content'])
        contexts.append(content)

    contexts, contexts_seg = process_input(contexts, segment=1)

    # break into 256 sized chunks
    inputs = []
    stride = 256
    for j, context in enumerate(contexts):
        chunks = [context[i:i + 256] for i in range(0, len(context), stride)]

        inp = []
        for chunk in chunks:
            sample = (torch.cat((torch.tensor(questions[j]), torch.tensor([102]), torch.tensor(chunk))))
            if len(sample) > 336:
                sample = sample[:336]
            else:
                sample = torch.cat((sample, torch.zeros(336 - len(sample))))
            inp.append(sample)
        inputs.append(torch.vstack(inp).int())

    bert_transformer.train()

    for epoch in range(10):
        for i, x in enumerate(inputs):
            torch.cuda.empty_cache()
            out = bert_transformer(x)

            # slide
            stride = 248
            out = [out[:, i:i + stride, :] for i in range(0, out.shape[1])]


train()
