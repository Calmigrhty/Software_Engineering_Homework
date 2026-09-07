import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

# TODO: 务必填入你的真实 Token
API_TOKEN = "Token"

# 初始化官方客户端，设置 5 分钟的超长超时以应对大模型的冷启动
client = InferenceClient(
    model="XLabs-AI/flux-RealismLora",
    token=API_TOKEN,
    timeout=300 
)

def generate_image(prompt):
    if not prompt.strip():
        raise gr.Error("提示词不能为空！")
    
    try:
        # 官方库的 text_to_image 方法会自动处理排队和后台加载问题
        image = client.text_to_image(prompt)
        return image
    except Exception as e:
        error_msg = str(e).lower()
        if "loading" in error_msg:
            raise gr.Error("服务器正在极速唤醒模型，请再等 30 秒后重新点击生成！")
        else:
            raise gr.Error(f"生成失败: {str(e)}")

with gr.Blocks(title="Flux 图像生成") as demo:
    gr.Markdown("### 软件工程作业：Flux RealismLora 图像交互生成")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                lines=5, 
                placeholder="请输入英文提示词，例如: A cinematic portrait of a young girl...", 
                label="Prompt (提示词)"
            )
            submit_btn = gr.Button("🚀 开始生成图像", variant="primary")
            
        with gr.Column():
            image_output = gr.Image(label="最终生成的图像", type="pil")
            
    submit_btn.click(fn=generate_image, inputs=prompt_input, outputs=image_output)

if __name__ == "__main__":
    demo.launch()
