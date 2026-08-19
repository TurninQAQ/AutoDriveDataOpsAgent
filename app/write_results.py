# /app/lib/result_writer.py (业务镜像内)
import json, tempfile, os
from datetime import datetime, timezone

def write_result(file_path: str, dataset: str, status: str, **kwargs):
    """原子写入JSON, 防止校验任务读到半写文件
        file_path: 保存 results.json 的路径, 需要放在数据集根目录下
        dataset: 数据集名字
    """
    dir_name = os.path.dirname(file_path)
    os.makedirs(dir_name, exist_ok=True)
    payload = {
        "dataset_path": dir_name,
        "dataset_name": dataset,
        "status": status,  # "success" | "failed"
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp_path, file_path)  # 原子rename
    except Exception:
        os.unlink(tmp_path)
        raise