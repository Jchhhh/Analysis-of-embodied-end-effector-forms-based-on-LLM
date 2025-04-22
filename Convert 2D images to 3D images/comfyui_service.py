import os
import json
import requests
import time
from typing import Optional
from pathlib import Path

class ComfyUIService:
    def __init__(self, 
                 server_url: str = "http://localhost:8188",
                 workflow_path: str = "ComfyUI-Trio.json"):
        """
        初始化 ComfyUI 服务
        
        Args:
            server_url: ComfyUI 服务器地址
            workflow_path: workflow JSON 文件路径
        """
        self.server_url = server_url
        self.workflow_path = workflow_path
        self.workflow = self._load_workflow()
        
    def _load_workflow(self) -> dict:
        """加载 workflow JSON 文件"""
        with open(self.workflow_path, 'r') as f:
            return json.load(f)
    
    def _get_node_id_by_title(self, title: str) -> Optional[str]:
        """根据节点标题获取节点ID"""
        for node_id, node_data in self.workflow.items():
            if isinstance(node_data, dict) and node_data.get('title') == title:
                return node_id
        return None
    
    def _upload_image(self, image_path: str) -> str:
        """上传图片到 ComfyUI 服务器"""
        url = f"{self.server_url}/upload/image"
        files = {'image': open(image_path, 'rb')}
        response = requests.post(url, files=files)
        return response.json()['name']
    
    def _queue_prompt(self, prompt: dict) -> str:
        """提交任务到 ComfyUI 服务器"""
        url = f"{self.server_url}/prompt"
        response = requests.post(url, json=prompt)
        return response.json()['prompt_id']
    
    def _get_history(self, prompt_id: str) -> dict:
        """获取任务历史"""
        url = f"{self.server_url}/history/{prompt_id}"
        response = requests.get(url)
        return response.json()
    
    def _download_output(self, filename: str, output_dir: str) -> str:
        """下载输出文件"""
        url = f"{self.server_url}/view?filename={filename}"
        response = requests.get(url)
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    
    def process_image(self, 
                     input_image_path: str,
                     output_dir: str = "output") -> str:
        """
        处理输入图片并生成 .glb 文件
        
        Args:
            input_image_path: 输入图片路径
            output_dir: 输出目录
            
        Returns:
            生成的 .glb 文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 上传输入图片
        uploaded_image = self._upload_image(input_image_path)
        
        # 更新 workflow 中的输入节点
        input_node_id = self._get_node_id_by_title("Input Image")
        if not input_node_id:
            raise ValueError("找不到输入图片节点")
        
        self.workflow[input_node_id]['inputs']['image'] = uploaded_image
        
        # 提交任务
        prompt_id = self._queue_prompt(self.workflow)
        
        # 等待任务完成
        while True:
            history = self._get_history(prompt_id)
            if prompt_id in history:
                if history[prompt_id]['status']['completed']:
                    break
                elif history[prompt_id]['status']['failed']:
                    raise RuntimeError("任务执行失败")
            time.sleep(1)
        
        # 获取输出文件
        output_node_id = self._get_node_id_by_title("Output GLB")
        if not output_node_id:
            raise ValueError("找不到输出节点")
        
        output_files = history[prompt_id]['outputs'][output_node_id]['images']
        if not output_files:
            raise RuntimeError("没有生成输出文件")
        
        # 下载输出文件
        output_path = self._download_output(output_files[0]['filename'], output_dir)
        return output_path

def main():
    # 示例用法
    service = ComfyUIService()
    try:
        # 使用当前目录下的 image.png 作为输入
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_image_path = os.path.join(current_dir, "image.png")
        
        # 检查文件是否存在
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"找不到输入图片文件: {input_image_path}")
            
        output_path = service.process_image(input_image_path)
        print(f"处理完成，输出文件保存在: {output_path}")
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main() 