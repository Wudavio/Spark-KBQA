import torch
from transformers import BertTokenizer
import bertcrf, ner_main
import os

def load_ner_model(config_file, model_name, label_num=3):
    model = bertcrf.BertCrf(config_name=config_file, num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(model_name))
    return model

def get_ner_model():
    ner_path = '/KgSpark/output/data2/'
    ner_processor = ner_main.preProcessorNer()
    ner_model = load_ner_model(config_file='/KgSpark/bert-base-chinese', \
        model_name=os.path.join(ner_path, 'pytorch_model.bin'), \
        label_num=len(ner_processor.get_labels()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model = ner_model.to(device)
    return ner_model

def get_entity(model, sentence) -> str:
    tokenizer = BertTokenizer.from_pretrained('/KgSpark/bert-base-chinese')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_length = 128
    sentence_list = list(sentence.strip().replace(' ',''))
    text = " ".join(sentence_list)

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # cls + ids + sep
    input_ids = tokenizer.build_inputs_with_special_tokens(ids)
    masks = [1] * len(input_ids)
    #token_type_ids = tokenizer.create_token_type_ids_from_sequences(idsA, idsB)
    token_type_ids = [0] * (len(ids) + 2)

    # pad seq to max length
    pad_seq = [0] * (max_seq_length - len(input_ids))
    input_ids += pad_seq
    masks += pad_seq
    token_type_ids += pad_seq

    assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
    assert len(masks) == max_seq_length, "Error with input length {} vs {}".format(len(masks), max_seq_length)
    assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(len(token_type_ids), max_seq_length)


    input_ids = torch.tensor(input_ids).reshape(1, -1).to(device)
    masks = torch.tensor(masks).reshape(1, -1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1, -1).to(device)

    model = model.to(device)
    model.eval()

    # when label is None, loss won't be computed
    output = model(input_ids = input_ids,
                tags = None,
                attention_mask = masks,
                token_type_ids = token_type_ids,
                decode=True)

    pred_tag = output[1][0]
    assert len(pred_tag) == len(sentence_list) or len(pred_tag) == max_len - 2

    pred_tag_len = len(pred_tag)

    CRF_LABELS = ['O', 'B-LOC', 'I-LOC']
    b_loc_idx = CRF_LABELS.index('B-LOC')
    i_loc_idx = CRF_LABELS.index('I-LOC')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pred_tag and i_loc_idx not in pred_tag:
        print("沒有在句子中[{}]中發現實體".format(sentence))
        return ''
    if b_loc_idx in pred_tag:
        entity_start_idx = pred_tag.index(b_loc_idx)
    else:
        entity_start_idx = pred_tag.index(i_loc_idx)

    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])

    for i in range(entity_start_idx+1, pred_tag_len):
        if pred_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break

    return "".join(entity_list)

