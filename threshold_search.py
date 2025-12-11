
import numpy as np, os, argparse, subprocess, json, time

def run_short_pdd(easy, medium, idx):
    ckpt_dir = f"results/threshold_tmp/run_{idx}_e{int(easy*100)}_m{int(medium*100)}"
    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = [
        "python","train_pdd.py",
        "--data_dir","./results/baseline",
        "--difficulty","./results/difficulty.npy",
        "--model","inception",
        "--epochs","5",
        "--batch_size","16",
        "--ckpt_dir", ckpt_dir,
        "--easy_pct", str(easy),
        "--medium_pct", str(medium),
        "--gamma","0.95",
        "--num_workers","2"
    ]
    print("Running:", " ".join(cmd))
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    took = time.time()-start
    out = r.stdout + "\n" + r.stderr
    # try to extract validation macro-F1 from output or result file
    val = None
    for line in out.splitlines():
        if "Val" in line and "macro" in line.lower():
            try:
                # fallback parsing: last float on line
                val = float(line.strip().split()[-1])
            except:
                pass
    # write log
    with open(os.path.join(ckpt_dir,"run_log.txt"),"w",encoding="utf8") as f:
        f.write(out)
    return {"easy":easy,"medium":medium,"val_metric":val if val is not None else None, "time":took, "ckpt_dir":ckpt_dir}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--easy_list", default="0.2,0.3,0.4,0.5")
    p.add_argument("--medium_list", default="0.6,0.7,0.8")
    p.add_argument("--out", default="./results/reports/threshold_results.json")
    args=p.parse_args()
    easy_list = [float(x) for x in args.easy_list.split(",")]
    medium_list = [float(x) for x in args.medium_list.split(",")]
    results=[]
    idx=0
    for e in easy_list:
        for m in medium_list:
            idx+=1
            if m<=e: continue
            r = run_short_pdd(e,m,idx)
            results.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        json.dump(results,f,indent=2)
    print("Saved threshold search results to", args.out)

if __name__=="__main__":
    main()
