import argparse

import torch
import torch.nn as nn

from transformers import AutoTokenizer
import OpenMatch as om
from tqdm import tqdm

def test(args, model, test_loader, device):
    rst_dict = {}
    model.eval()
    for test_batch in tqdm(test_loader):
        query_id, doc_id, retrieval_score = test_batch['query_id'], test_batch['doc_id'], test_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device), test_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(test_batch['query_wrd_idx'].to(device), test_batch['query_wrd_mask'].to(device),
                                       test_batch['doc_wrd_idx'].to(device), test_batch['doc_wrd_mask'].to(device),
                                       test_batch['query_ent_idx'].to(device), test_batch['query_ent_mask'].to(device),
                                       test_batch['doc_ent_idx'].to(device), test_batch['doc_ent_mask'].to(device),
                                       test_batch['query_des_idx'].to(device), test_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(test_batch['query_idx'].to(device), test_batch['query_mask'].to(device),
                                       test_batch['doc_idx'].to(device), test_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s) in zip(query_id, doc_id, batch_score):
                if q_id in rst_dict:
                    rst_dict[q_id].append((b_s, d_id))
                else:
                    rst_dict[q_id] = [(b_s, d_id)]
    return rst_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-test', action=om.utils.DictOrStr, default='./data/test_toy.jsonl')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=32)
    parser.add_argument('-max_doc_len', type=int, default=256)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument("--checkpoint_no", type=int, default=-1)
    args = parser.parse_args()

    args.model = args.model.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    final_result = {}

    for fold in range(5):
        print('reading test data...')
        test_set = om.data.datasets.BertDataset(
            dataset=args.test + "." + str(fold),
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

        test_loader = om.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8
        )

        model = om.models.Bert(
            pretrained=args.pretrain,
            mode=args.mode,
            task=args.task
        )

        state_dict = torch.load(args.checkpoint + "_" + str(fold) if args.checkpoint_no == -1 else args.checkpoint + "_" + str(fold) + "_step" + str(args.checkpoint_no) + ".bin")
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        rst_dict = test(args, model, test_loader, device)
        final_result.update(rst_dict)

        del model
        torch.cuda.empty_cache()
        
    om.utils.save_trec(args.res, final_result)

if __name__ == "__main__":
    main()
