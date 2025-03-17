import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import *

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'
        print(f'eval model load: {ckp}')

        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            './MiniMind2',
            trust_remote_code=True
        )
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '二氧化碳在空气中',
            '杭州市的美食有',
            '陈述：\n在过去的几十年里，全球气候变化问题日益严重，科学家们通过各种研究和数据分析，揭示了人类活动对地球气候系统的深远影响。工业化进程中的大量温室气体排放，尤其是二氧化碳和甲烷，导致了全球气温的显著上升。这种气温上升不仅引发了极端天气事件的频繁发生，如热浪、暴雨、干旱和飓风，还对生态系统、农业、水资源和人类健康产生了广泛的影响。极地冰盖的融化导致海平面上升，威胁着沿海城市和岛屿国家的生存。此外，气候变化还加剧了社会不平等，因为贫困国家和社区往往缺乏应对气候灾害的资源。国际社会已经意识到这一问题的紧迫性，并通过《巴黎协定》等全球性协议试图减缓气候变化的影响。然而，尽管各国承诺减少碳排放，实际执行力度仍然不足，全球气温上升的趋势尚未得到有效遏制。与此同时，科学家们也在积极探索应对气候变化的创新解决方案，如碳捕获技术、可再生能源的广泛应用以及生态恢复项目。公众意识的提高和个体行为的改变也被视为应对气候变化的重要组成部分。然而，面对这一全球性挑战，国际合作、政策执行和技术创新的结合仍然是解决问题的关键。\n\n问题：\n\n全球气候变化对人类和地球生态系统的影响有哪些具体表现？国际社会在应对气候变化方面采取了哪些措施？这些措施的效果如何？未来应对气候变化的挑战和可能的解决方案是什么？\n\n答案：',
            '<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s> <s>打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。</s> <s>为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。</s> <s>非常感谢你的回答。请告诉我，这些数据是关于',
            '<s>最好是一些日常用语和工作相关的单词。好的，我可以为您提供一些相关单词和短语的列表。您想要我立即发送吗？</s> <s>你好，能告诉我世界杯的冠军是谁吗？你好！当然可以，最近一届世界杯的冠军是法国。还有什么我能帮助你的吗？</s> <s>为以下这个名字提供三个相应的同义词。\n名字: 韦恩相应的同义词：韦恩可以被替换为韦尔纳，韦恩特，维恩。</s> <s>接下来，请你找到这段文本中出现次数最少的单词。示例文本中出现次数最少的单词是"检测"，它只出现了一次。</s> <s>听起来很不错，我会去看看这部电影的',
            '<s>下文中存在哪些实体：\n安徽文化音像出版社自1995年以来，合作出版引进文艺节目24个。安徽文化音像出版社</s> <s>下文中存在哪些实体：\n因此，企业在提高产品质量的同时，也应切实采取措施，提高产品说明书的质量。不存在实体</s> <s>实体抽取：\n他还给俄共领导人久加诺夫打了电话，希望俄共“多考虑国家利益，少考虑党派利益”。久加诺夫,俄共</s> <s>实体抽取：\n医院除招收一批著名老专家外，还吸收了近年学成回国的一些医药学博士、硕士等专业人员。组织：医院</s> <s>抽取出下文中的实体：\n我天',
            '<s>请提取以下文本中的人物和组织名称：张三和李四在百度和腾讯工作。\n人物名称：张三、李四\n组织名称：百度、腾讯</s> <s>将这个句子改写成简洁的版本。\n我想要今天晚上吃中餐，因为我已经吃惯了西餐了。我今晚想吃中餐，吃西餐吃腻了。</s> <s>重写以下句子，使其更具有说服力。\n治疗头痛的这种药物被证明是有效的。研究证实，这种药物可以有效地缓解头痛。</s> <s>对于给定的文本片段，重新编写以更加简练地表示意思。\n此时此刻，我正在考虑一个例子。现在，我在考虑一个例子。</s> <s>根据以下输入，生成',
            '<s>请提取以下文本中的人物和组织名称：张三和李四在百度和腾讯工作。\n人物名称：张三、李四\n组织名称：百度、腾讯</s> <s>将这个句子改写成简洁的版本。\n我想要今天晚上吃中餐，因为我已经吃惯了西餐了。我今晚想吃中餐，吃西餐吃腻了。</s> <s>重写以下句子，使其更具有说服力。\n治疗头痛的这种药物被证明是有效的。研究证实，这种药物可以有效地缓解头痛。</s> <s>对于给定的文本片段，重新编写以更加简练地表示意思。\n此时此刻，我正在考虑一个例子。现在，我在考虑一个例子。</s> <s>根据以下输入，生成相应的语句，且这些语句应与输入意思相反。\n这个电影非常有趣。这个电影没有一点好笑的地方。</s> <s>从以下文本中提取金额\n我在商店里花了50美元买了一件衬衫。从上面的文本中，金额为50美元，可以被提取出来。</s> <s>用您自己的语言重新表述以下句子。\n机器翻译可以帮助人们跨越语言障碍。机器翻译技术可以协助人们克服语言障碍。</s> <s>将以下句子精简成5个单词的版本。\n如果我',
            '地球上最大的动物有',
            '世界上最高的山峰是',
            '你是一个小说写手，你现在需要写一个400字左右的童话故事概要。故事概要为：',
            '你是一个python专家，你先需要写一个最小子序列和的面试题，给出测试数据以及封装为函数的最小子序列和实现。代码实现为：',
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解“大语言模型”这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.',
                '地球上最大的动物有',
                '世界上最高的山峰是',
                '你是一个小说写手，你现在需要写一个400字左右的童话故事概要。故事概要为：',
                '你是一个python专家，你先需要写一个最小子序列和的面试题，给出测试数据以及封装为函数的最小子序列和实现。代码实现为：',
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # 此处max_seq_len（最大允许输入长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    # MiniMind2-moe (145M)：(dim=640, n_layers=8, use_moe=True)
    # MiniMind2-Small (26M)：(dim=512, n_layers=8)
    # MiniMind2 (104M)：(dim=768, n_layers=16)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    parser.add_argument('--test_mode', default=0, type=int, help="[0] 自动测试, [1] 手动输入")
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'eval_model args unknow: {unknown}')
    return args


def main(args):
    print(f'args: {args}')
    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    # test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    test_mode = args.test_mode
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})

    del model, tokenizer

if __name__ == "__main__":
    args = get_args()
    main(args)
