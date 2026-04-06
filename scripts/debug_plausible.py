import asyncio
import json
from src.extract_references.clients import AnystyleClient
from src.extract_references.heuristics import CitationParserEngine

async def test():
    client = AnystyleClient()
    engine = CitationParserEngine()
    
    # R9 from paper 2602.23452v1
    raw = "Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 [cs.CV] https://arxiv.org/abs/2511.21631"
    
    anystyle_res = await client.parse(raw)
    parsed = engine.digest_anystyle_json(raw, anystyle_res)
    
    print(f"Parsed Title: {parsed.get('title')}")
    print(f"Parsed Year: {parsed.get('year')}")
    print(f"Fields Present: {sum(1 for k in ['authors', 'title', 'venue', 'year'] if parsed.get(k)) + (1 if parsed.get('doi') or parsed.get('arxiv_id') or parsed.get('url') else 0)}")
    print(f"Is Plausible: {engine.is_plausible_reference(raw, parsed)}")
    print(f"Full parsed: {json.dumps(parsed, indent=2)}")

if __name__ == '__main__':
    asyncio.run(test())
