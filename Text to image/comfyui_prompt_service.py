import os
import json
import requests
import time
from typing import Optional
from pathlib import Path

class ComfyUIPromptService:
    def __init__(self, 
                 server_url: str = "http://localhost:8188",
                 workflow_path: str = "ComfyUI-Prompt.json"):
        """
        初始化 ComfyUI Prompt 服务
        
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
    
    def _download_output(self, filename: str, output_path: str) -> str:
        """下载输出文件"""
        url = f"{self.server_url}/view?filename={filename}"
        response = requests.get(url)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    
    def generate_image(self, 
                      prompt: str,
                      output_path: str = "image.png") -> str:
        """
        根据 prompt 生成图片
        
        Args:
            prompt: 输入的提示词
            output_path: 输出图片路径
            
        Returns:
            生成的图片文件路径
        """
        # 更新 workflow 中的 prompt 节点
        prompt_node_id = self._get_node_id_by_title("Text Prompt")
        if not prompt_node_id:
            raise ValueError("找不到提示词输入节点")
        
        self.workflow[prompt_node_id]['inputs']['text'] = prompt
        
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
        output_node_id = self._get_node_id_by_title("Output Image")
        if not output_node_id:
            raise ValueError("找不到输出节点")
        
        output_files = history[prompt_id]['outputs'][output_node_id]['images']
        if not output_files:
            raise RuntimeError("没有生成输出文件")
        
        # 下载输出文件
        output_path = self._download_output(output_files[0]['filename'], output_path)
        return output_path

def main():
    # 示例用法
    service = ComfyUIPromptService()
    
    try:
        # 从终端获取用户输入的 prompt
        print("请输入提示词（输入完成后按回车键）：")
        prompt = input().strip()
        
        if not prompt:
            raise ValueError("提示词不能为空")
            
        # 生成图片
        output_path = service.generate_image(prompt)
        print(f"图片生成完成，保存为: {output_path}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main() 