# M2 运行方法

在项目根目录安装依赖，然后进入 `M2`：

```powershell
python -m pip install -r ..\requirements.txt
python run.py fetch
python run.py prepare
python run.py run --model-config model_config.example.json
```

使用 `--limit` 限制样本数，使用 `--methods` 指定方法，使用 `--run-dir` 指定输出目录：

```powershell
python run.py run --model-config model_config.example.json --methods ours --limit 20 --run-dir results
```

模型服务地址、模型名和 API 密钥环境变量名在 `model_config.example.json` 中配置。
