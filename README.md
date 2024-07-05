# VT-Trans

**Python Version**: Python>=3.6

**Package Requirements**: torch>=1.4.0 tensorboardX numpy>=1.19.0

Before running the scripts, please install fairseq dependencies by:
```
    pip install --editable .
```

* Step 1: Combine sentences to create context-aware information:
```
    mkdir exp_mbart
    k=3: bash exp_gtrans/run-all.sh prepare-mbart exp_mbart 3 
    k=5: bash exp_gtrans/run-all.sh prepare-mbart exp_mbart 5
    k=7: bash exp_gtrans/run-all.sh prepare-mbart exp_mbart 7
```

* Step 2: Prepare data: 
```
    bash exp_gtrans/run-all.sh prepare-mbart exp_mbart
```

* Step 3: Train model:
```
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp_gtrans/run-all.sh run-mbart train exp_mbart
```

* Step 4: Evaluate model:
```
    bash exp_gtrans/run-all.sh run-mbart test exp_mbart
```

Some code are borrowed from [G-Transformer](https://github.com/baoguangsheng/g-transformer). Thanks for their work.