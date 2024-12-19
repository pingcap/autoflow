from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.models import Upload
from app.rag.datasource import FileDataSource
from app.file_storage import default_file_storage

router = APIRouter()

@router.get("/documents/{document_id}/download")
def download_file(document_id: int):
    isfound = False    
    for f_config in FileDataSource.config:
        if f_config["file_id"] == document_id:
            isfound = True
    # 找到了就返回文件
    if isfound == True:
        upload = FileDataSource.session.get(Upload, document_id)
        return FileResponse(path=upload.path, filename=upload.name, media_type='application/octet-stream')
    # 没找到应该 302 到对应的 url，但先404
    else:
        raise HTTPException(status_code=404, detail="lmw : File not found")


