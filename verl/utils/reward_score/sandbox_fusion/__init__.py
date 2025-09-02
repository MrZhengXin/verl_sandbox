# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import traceback
import re

# from .utils import check_correctness
from .utils_common_evaluate import check_correctness_common_evaluate

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

def compute_score(sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_cases, continuous=False, timeout=30):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    # remove <think>.*</think> tags if they exist
    completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL).strip()
    solution = completion
    language_str = re.search(r'```(\w+)', completion)
    if language_str:
        language = language_str.group(1).strip()
    else:
        # Default to Python if no language is specified
        language = "python"
        
    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if not test_cases or "input" not in test_cases or "output" not in test_cases:
            logger.error("Invalid test_cases structure.")
            logger.error(f"{test_cases}")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        res_list, metadata_list = check_correctness_common_evaluate(sandbox_fusion_url=sandbox_fusion_url, in_outs=test_cases, generation=solution, timeout=timeout, concurrent_semaphore=concurrent_semaphore, language=language)

        # Calculate score
        if not res_list:  # If there are no results (e.g., invalid input)
            return 0.0, metadata_list

        score = metadata_list[0].get("score", 0.0)  # Default to 0.0 if score not found
        final_metadata = metadata_list

    except Exception as e:
        # logger.error(f"Error during compute_score: {e}")
        # traceback.print_exc()
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Ensure float and list are returned
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]


if False and __name__ == "__main__":
    # Example usage
    sandbox_fusion_url = "http://10.0.1.5:8080/common_evaluate_batch"
    concurrent_semaphore = None  # Replace with actual semaphore if needed
    from datasets import load_dataset
    data_path = "/mnt/data1/zhengxin2020/verl/data/verifiable-coding-problems-python-only/train.parquet"
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    instance = dataset[0]
    # print(instance)
    # input()
    # {'prompt': "Solve the following coding problem using the programming language python:\n\nThere are some websites that are accessible through several different addresses. For example, for a long time Codeforces was accessible with two hostnames codeforces.com and codeforces.ru.\n\nYou are given a list of page addresses being queried. For simplicity we consider all addresses to have the form http://<hostname>[/<path>], where:\n\n  <hostname>\xa0— server name (consists of words and maybe some dots separating them),  /<path>\xa0— optional part, where <path> consists of words separated by slashes. \n\nWe consider two <hostname> to correspond to one website if for each query to the first <hostname> there will be exactly the same query to the second one and vice versa\xa0— for each query to the second <hostname> there will be the same query to the first one. Take a look at the samples for further clarifications.\n\nYour goal is to determine the groups of server names that correspond to one website. Ignore groups consisting of the only server name.\n\nPlease note, that according to the above definition queries http://<hostname> and http://<hostname>/ are different.\n\n\n-----Input-----\n\nThe first line of the input contains a single integer n (1 ≤ n ≤ 100 000)\xa0— the number of page queries. Then follow n lines each containing exactly one address. Each address is of the form http://<hostname>[/<path>], where:\n\n  <hostname> consists of lowercase English letters and dots, there are no two consecutive dots, <hostname> doesn't start or finish with a dot. The length of <hostname> is positive and doesn't exceed 20.  <path> consists of lowercase English letters, dots and slashes. There are no two consecutive slashes, <path> doesn't start with a slash and its length doesn't exceed 20. \n\nAddresses are not guaranteed to be distinct.\n\n\n-----Output-----\n\nFirst print k\xa0— the number of groups of server names that correspond to one website. You should count only groups of size greater than one.\n\nNext k lines should contain the description of groups, one group per line. For each group print all server names separated by a single space. You are allowed to print both groups and names inside any group in arbitrary order.\n\n\n-----Examples-----\nInput\n10\nhttp://abacaba.ru/test\nhttp://abacaba.ru/\nhttp://abacaba.com\nhttp://abacaba.com/test\nhttp://abacaba.de/\nhttp://abacaba.ru/test\nhttp://abacaba.de/test\nhttp://abacaba.com/\nhttp://abacaba.com/t\nhttp://abacaba.com/test\n\nOutput\n1\nhttp://abacaba.de http://abacaba.ru \n\nInput\n14\nhttp://c\nhttp://ccc.bbbb/aba..b\nhttp://cba.com\nhttp://a.c/aba..b/a\nhttp://abc/\nhttp://a.c/\nhttp://ccc.bbbb\nhttp://ab.ac.bc.aa/\nhttp://a.a.a/\nhttp://ccc.bbbb/\nhttp://cba.com/\nhttp://cba.com/aba..b\nhttp://a.a.a/aba..b/a\nhttp://abc/aba..b/a\n\nOutput\n2\nhttp://cba.com http://ccc.bbbb \nhttp://a.a.a http://a.c http://abc\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.", 'solutions': ['```python\n# Bartek Kostka\n#  You are not prepared!\n\n#include "bits/stdc++.h"\n\nn = int(input())\nW = {}\nfor i in range(n):\n    adr = input()\n    adr = adr.split("/")\n    if adr[-1] == \'\':\n        adr[-1] = \'?\'\n    domena = "/".join(adr[:3])\n    adres = "/".join(adr[3:])\n    #print(domena, adres)\n    if domena not in W:\n        W[domena] = set()\n    W[domena].add(adres)\n\nE = {}\nfor key, ele in list(W.items()):\n    #print(key, ele)\n    lele = "#".join(sorted(list(ele)))\n    if lele not in E:\n        E[lele] = []\n    E[lele].append(key)\n\nres = 0\nfor key, ele in list(E.items()):\n    if len(ele) > 1:\n        res += 1\n\nprint(res)\nfor key, ele in list(E.items()):\n    if len(ele) > 1:\n        print(" ".join(ele))\n\n```'], 'reward_model': {'ground_truth': '{"inputs": ["10\\nhttp://abacaba.ru/test\\nhttp://abacaba.ru/\\nhttp://abacaba.com\\nhttp://abacaba.com/test\\nhttp://abacaba.de/\\nhttp://abacaba.ru/test\\nhttp://abacaba.de/test\\nhttp://abacaba.com/\\nhttp://abacaba.com/t\\nhttp://abacaba.com/test\\n", "14\\nhttp://c\\nhttp://ccc.bbbb/aba..b\\nhttp://cba.com\\nhttp://a.c/aba..b/a\\nhttp://abc/\\nhttp://a.c/\\nhttp://ccc.bbbb\\nhttp://ab.ac.bc.aa/\\nhttp://a.a.a/\\nhttp://ccc.bbbb/\\nhttp://cba.com/\\nhttp://cba.com/aba..b\\nhttp://a.a.a/aba..b/a\\nhttp://abc/aba..b/a\\n", "10\\nhttp://tqr.ekdb.nh/w\\nhttp://p.ulz/ifw\\nhttp://w.gw.dw.xn/kpe\\nhttp://byt.mqii.zkv/j/xt\\nhttp://ovquj.rbgrlw/k..\\nhttp://bv.plu.e.dslg/j/xt\\nhttp://udgci.ufgi.gwbd.s/\\nhttp://l.oh.ne.o.r/.vo\\nhttp://l.oh.ne.o.r/w\\nhttp://tqr.ekdb.nh/.vo\\n", "12\\nhttp://ickght.ck/mr\\nhttp://a.exhel/.b\\nhttp://a.exhel/\\nhttp://ti.cdm/\\nhttp://ti.cdm/x/wd/lm.h.\\nhttp://ickght.ck/a\\nhttp://ickght.ck\\nhttp://c.gcnk.d/.b\\nhttp://c.gcnk.d/x/wd/lm.h.\\nhttp://ti.cdm/.b\\nhttp://a.exhel/x/wd/lm.h.\\nhttp://c.gcnk.d/\\n", "14\\nhttp://jr/kgb\\nhttp://ps.p.t.jeua.x.a.q.t\\nhttp://gsqqs.n/t/\\nhttp://w.afwsnuc.ff.km/cohox/u.\\nhttp://u.s.wbumkuqm/\\nhttp://u.s.wbumkuqm/cohox/u.\\nhttp://nq.dzjkjcwv.f.s/bvm/\\nhttp://zoy.shgg\\nhttp://gsqqs.n\\nhttp://u.s.wbumkuqm/b.pd.\\nhttp://w.afwsnuc.ff.km/\\nhttp://w.afwsnuc.ff.km/b.pd.\\nhttp://nq.dzjkjcwv.f.s/n\\nhttp://nq.dzjkjcwv.f.s/ldbw\\n", "15\\nhttp://l.edzplwqsij.rw/\\nhttp://m.e.mehd.acsoinzm/s\\nhttp://yg.ttahn.xin.obgez/ap/\\nhttp://qqbb.pqkaqcncodxmaae\\nhttp://lzi.a.flkp.lnn.k/o/qfr.cp\\nhttp://lzi.a.flkp.lnn.k/f\\nhttp://p.ngu.gkoq/.szinwwi\\nhttp://qqbb.pqkaqcncodxmaae/od\\nhttp://qqbb.pqkaqcncodxmaae\\nhttp://wsxvmi.qpe.fihtgdvi/e./\\nhttp://p.ngu.gkoq/zfoh\\nhttp://m.e.mehd.acsoinzm/xp\\nhttp://c.gy.p.h.tkrxt.jnsjt/j\\nhttp://wsxvmi.qpe.fihtgdvi/grkag.z\\nhttp://p.ngu.gkoq/t\\n", "15\\nhttp://w.hhjvdn.mmu/.ca.p\\nhttp://m.p.p.lar/\\nhttp://lgmjun.r.kogpr.ijn/./t\\nhttp://bapchpl.mcw.a.lob/d/ym/./g.q\\nhttp://uxnjfnjp.kxr.ss.e.uu/jwo./hjl/\\nhttp://fd.ezw.ykbb.xhl.t/\\nhttp://i.xcb.kr/.ca.p\\nhttp://jofec.ry.fht.gt\\nhttp://qeo.gghwe.lcr/d/ym/./g.q\\nhttp://gt\\nhttp://gjvifpf.d/d/ym/./g.q\\nhttp://oba\\nhttp://rjs.qwd/v/hi\\nhttp://fgkj/\\nhttp://ivun.naumc.l/.ca.p\\n", "20\\nhttp://gjwr/xsoiagp/\\nhttp://gdnmu/j\\nhttp://yfygudx.e.aqa.ezh/j\\nhttp://mpjxue.cuvipq/\\nhttp://a/\\nhttp://kr/..n/c.\\nhttp://a/xsoiagp/\\nhttp://kr/z\\nhttp://kr/v.cv/rk/k\\nhttp://lvhpz\\nhttp://qv.v.jqzhq\\nhttp://y.no/\\nhttp://kr/n\\nhttp://y.no/xsoiagp/\\nhttp://kr/ebe/z/\\nhttp://olsvbxxw.win.n/j\\nhttp://p.ct/j\\nhttp://mpjxue.cuvipq/xsoiagp/\\nhttp://kr/j\\nhttp://gjwr/\\n", "1\\nhttp://a\\n", "1\\nhttp://a.a.a.f.r.f.q.e.w.a/fwe..sdfv....\\n", "3\\nhttp://abacaba.com/test\\nhttp://abacaba.de/test\\nhttp://abacaba.de/test\\n"], "outputs": ["1\\nhttp://abacaba.de http://abacaba.ru \\n", "2\\nhttp://cba.com http://ccc.bbbb \\nhttp://a.a.a http://a.c http://abc \\n", "2\\nhttp://l.oh.ne.o.r http://tqr.ekdb.nh \\nhttp://bv.plu.e.dslg http://byt.mqii.zkv \\n", "1\\nhttp://a.exhel http://c.gcnk.d http://ti.cdm \\n", "2\\nhttp://ps.p.t.jeua.x.a.q.t http://zoy.shgg \\nhttp://u.s.wbumkuqm http://w.afwsnuc.ff.km \\n", "0\\n", "4\\nhttp://gt http://jofec.ry.fht.gt http://oba \\nhttp://fd.ezw.ykbb.xhl.t http://fgkj http://m.p.p.lar \\nhttp://i.xcb.kr http://ivun.naumc.l http://w.hhjvdn.mmu \\nhttp://bapchpl.mcw.a.lob http://gjvifpf.d http://qeo.gghwe.lcr \\n", "3\\nhttp://lvhpz http://qv.v.jqzhq \\nhttp://a http://gjwr http://mpjxue.cuvipq http://y.no \\nhttp://gdnmu http://olsvbxxw.win.n http://p.ct http://yfygudx.e.aqa.ezh \\n", "0\\n", "0\\n", "1\\nhttp://abacaba.com http://abacaba.de \\n"]}', 'style': 'rule'}, 'data_source': 'apps'}
    test_cases = json.loads(instance['reward_model']['ground_truth'])
    print(test_cases)
    print("input" not in test_cases or "output" not in test_cases)
    completion = instance['solutions'][0]
    # completion = "<think>\nhahahah\n</think>```python\n\nn = int(input())\na = list(map(int, input().split()))\n\n# Initialize DP: list of tuples (value, cost)\ndp = [(a[0], 0)]\n\nfor i in range(1, n):\n    new_dp = []\n    current = a[i]\n    for (prev_val, prev_cost) in dp:\n        if current > prev_val + 1:\n            new_val = current\n            new_cost = prev_cost\n        else:\n            new_val = prev_val + 1\n            new_cost = prev_cost + (new_val - current)\n        new_dp.append((new_val, new_cost))\n    \n    # Sort the new_dp by value\n    new_dp.sort()\n    \n    # Process to keep only the minimal cost entries\n    min_cost = float('inf')\n    processed = []\n    for val, cost in new_dp:\n        if cost < min_cost:\n            processed.append((val, cost))\n            min_cost = cost\n    dp = processed\n\n# The minimal cost is the minimal in the last dp list\nif dp:\n    print(min(dp, key=lambda x: x[1])[1])\nelse:\n    print(0)\n\n```"
    # test_cases = {
    #     'fn_name': None, 
    #     'input': ['7\n2 1 5 11 5 9 11', '5\n5 4 3 2 1', '2\n1 1000', '2\n1000 1', '5\n100 80 60 70 90', '10\n10 16 17 11 1213 1216 1216 1209 3061 3062', '20\n103 103 110 105 107 119 113 121 116 132 128 124 128 125 138 137 140 136 154 158', '1\n1', '5\n1 1 1 2 3', '1\n1000', '50\n499 780 837 984 481 526 944 482 862 136 265 605 5 631 974 967 574 293 969 467 573 845 102 224 17 873 648 120 694 996 244 313 404 129 899 583 541 314 525 496 443 857 297 78 575 2 430 137 387 319', '75\n392 593 98 533 515 448 220 310 386 79 539 294 208 828 75 534 875 493 94 205 656 105 546 493 60 188 222 108 788 504 809 621 934 455 307 212 630 298 938 62 850 421 839 134 950 256 934 817 209 559 866 67 990 835 534 672 468 768 757 516 959 893 275 315 692 927 321 554 801 805 885 12 67 245 495', '10\n26 723 970 13 422 968 875 329 234 983', '20\n245 891 363 6 193 704 420 447 237 947 664 894 512 194 513 616 671 623 686 378', '5\n850 840 521 42 169'], 
    #     'output': ['9', '12', '0', '1000', '54', '16', '43', '0', '3', '0', '12423', '17691', '2546', '3208', '1485'], 
    #     'type': 'stdin_stdout'
    # }
    continuous = False
    timeout = 30
    score, metadata = compute_score(
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
        completion=completion,
        test_cases=test_cases,
        continuous=continuous,
        timeout=timeout
    )
    print(f"Score: {score}")
    print(f"Metadata: {metadata}")
    