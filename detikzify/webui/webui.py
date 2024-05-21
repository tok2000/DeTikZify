from functools import cache, lru_cache, partial
from inspect import signature
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from os import fdopen
from sys import float_info
from tempfile import TemporaryDirectory, mkstemp
from time import time
from typing import Dict

from PIL import Image
import gradio as gr
from torch import bfloat16, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from transformers import TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available

from ..infer import DetikzifyPipeline, TikzDocument
from ..model import load
from .strings import ALGORITHMS, BANNER, CSS, GALLERY_DESELECT_HACK, MODELS


def to_svg(
    tikzdoc: TikzDocument,
    build_dir: str
):
    if not tikzdoc.is_rasterizable:
        if tikzdoc.compiled_with_errors:
            raise gr.Error("TikZ code did not compile!")
        else:
            gr.Warning("TikZ code compiled to an empty image!")
    elif tikzdoc.compiled_with_errors:
        gr.Warning("TikZ code compiled with errors!")

    fd, path = mkstemp(dir=build_dir, suffix=".svg")
    with fdopen(fd, "w") as f:
        if pdf:=tikzdoc.pdf:
            f.write(pdf[0].get_svg_image())
    return path if pdf else None

# https://stackoverflow.com/a/50992575
def make_ordinal(n):
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

class MctsOutputs(set):
    def __init__(self, build_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_dir, self.svgmap, self.fails = build_dir, dict(), 0

    def add(self, score, tikzdoc): # type: ignore
        if (score, tikzdoc) not in self:
            try:
                 svg = to_svg(tikzdoc, build_dir=self.build_dir)
                 super().add((score, tikzdoc))
                 self.svgmap[tikzdoc] = svg
            except gr.Error:
                gr.Warning("TikZ code did not compile, discarding output!")
                if len(self): self.fails += 1
        elif len(self): self.fails += 1

    @property
    def programs(self):
        return [tikzdoc.code for _, tikzdoc in sorted(self, key=itemgetter(0), reverse=True)]

    @property
    def images(self):
        return [
            (self.svgmap[tikzdoc], make_ordinal(idx))
            for idx, (_, tikzdoc) in enumerate(sorted(self, key=itemgetter(0), reverse=True), 1)
        ]

    @property
    def first_success(self):
        return len(self) == 1 and not self.fails

@lru_cache(maxsize=1)
def cached_load(*args, **kwargs):
    gr.Info("Instantiating model. This could take a while...")
    return load(*args, **kwargs)

def inference(
    model_name: str,
    image: Dict[str, Image.Image],
    temperature: float,
    top_p: float,
    top_k: int,
    mcts_exploration: float,
    mcts_timeout: int,
    mcts_strict: bool,
    preprocess: bool,
    algorithm: str,
    compile_timeout: int,
):
    model, tokenizer = cached_load(
        base_model=model_name,
        device_map="auto",
        torch_dtype=bfloat16 if is_cuda_available() and is_bf16_supported() else float16,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    )
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer.text, # type: ignore
        skip_special_tokens=True
    )
    generate = DetikzifyPipeline(
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        temperature=temperature + float_info.epsilon,
        top_p=top_p,
        top_k=top_k,
        compile_timeout=compile_timeout,
    )
    thread = ThreadPool(processes=1)

    @cache
    def compile_info_once():
        gr.Info('Compiling, please wait.')

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        if algorithm == "mcts":
            iterator = generate.simulate(
                image=image['composite'],
                exploration=mcts_exploration,
                strict=mcts_strict,
                preprocess=preprocess,
            )
            # we have to implement our own timer since we use threading and would run into a deadlock otherwise
            tikzdocs, start = MctsOutputs(build_dir=tmpdir), time()
            while True:
                code, async_result = "", thread.apply_async(func=lambda: next(iterator))
                for new_text in streamer:
                    code += new_text
                    yield code, tikzdocs.programs, None, tikzdocs.images, gr.Tabs()
                compile_info_once()
                tikzdocs.add(*async_result.get())
                yield code, tikzdocs.programs, None, tikzdocs.images, gr.Tabs(selected=1 if tikzdocs.first_success else None)
                if time() - start > mcts_timeout * 60:
                    yield "", tikzdocs.programs, None, tikzdocs.images, gr.Tabs(selected=1)
                    break
        else: # sampling
            code, async_result = "", thread.apply_async(
                func=generate.sample,
                kwds=dict(image=image['composite'], preprocess=preprocess)
            )
            for new_text in streamer:
                code += new_text
                yield code, [], None, None, gr.Tabs()
            compile_info_once()
            tikzdoc = async_result.get()
            yield tikzdoc.code, [], to_svg(tikzdoc, build_dir=tmpdir), None, gr.Tabs(selected=1)

def check_inputs(image: Dict[str, Image.Image]):
    if image['composite'].getcolors(1) is not None:
        raise gr.Error("Image has no content!")

def make_light(stylable):
    """
    Patch gradio to only contain light mode colors.
    """
    if isinstance(stylable, gr.themes.Base): # remove dark variants from the entire theme
        params = signature(stylable.set).parameters
        colors = {color: getattr(stylable, color.removesuffix("_dark")) for color in dir(stylable) if color in params}
        return stylable.set(**colors)
    elif isinstance(stylable, gr.Blocks): # also handle components which do not use the theme (e.g. modals)
        stylable.load(
            fn=None,
            js="() => document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'))"
        )
        return stylable
    else:
        raise ValueError

def build_ui(
    model=list(MODELS)[0],
    lock=False,
    light=False,
    lock_reason="Duplicate this space to be able to change this value.",
    timeout=60,
    algorithm=list(ALGORITHMS)[0]
):
    theme = make_light(gr.themes.Soft()) if light else gr.themes.Soft()
    with gr.Blocks(css=CSS, theme=theme, title="DeTikZify", head=GALLERY_DESELECT_HACK) as demo: # type: ignore
        if light: make_light(demo)
        gr.HTML(BANNER)
        with gr.Row(variant="panel"):
            with gr.Column():
                sketchpad = gr.ImageEditor(
                    sources=["upload", "clipboard"],
                    elem_classes="input-image",
                    type="pil",
                    label="Sketchpad",
                    show_label=False,
                    canvas_size=(492, 492),
                    brush=gr.Brush(
                        colors=["black", "red", "green", "blue"],
                        default_size=1
                    ),
                )
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.ClearButton(sketchpad, variant="stop")
            with gr.Column(elem_classes="outputs"):
                with gr.Tabs() as tabs:
                    with gr.TabItem(label:="TikZ Code", id=0):
                        stream_code = gr.Code(label=label, show_label=False, elem_classes="output-code", elem_id="stream-code")
                        gallery_code = gr.Code(label=label, show_label=False, visible=False, elem_classes="output-code")
                    with gr.TabItem(label:="Compiled Images", id=1):
                        result_gallery = gr.Gallery(
                            label=label,
                            show_label=False,
                            show_share_button=False,
                            elem_classes="output-image",
                            visible=algorithm=="mcts",
                            columns=3,
                        )
                        # see strings.py how preview-close is used to detect closing of preview
                        preview_close_btn = gr.Button(visible=False, elem_id="preview-close")
                        result_image = gr.Image(
                            label=label,
                            show_label=False,
                            show_share_button=False,
                            elem_classes="output-image",
                            visible=algorithm=="sampling",
                        )
                    clear_btn.add([stream_code, gallery_code, result_image, result_gallery])
        with gr.Accordion(label="Settings", open=False):
            gr.Markdown(
                "For additional information and usage tips check out our [paper](TODO) and the "
                "[documentation](https://github.com/potamides/DeTikZify/tree/main/detikzify/webui)."
            )
            base_model_info="Which DeTikZify model to use for inference.",
            base_model = gr.Dropdown(
                label="Base Model",
                allow_custom_value=True,
                info=f"{lock_reason} {base_model_info}" if lock else base_model_info, # type: ignore
                choices=list(({model: model} | MODELS).items()),
                value=MODELS.get(model, model),
                interactive=not lock,
            )
            algorithm_info = (
                'Whether to use Monte Caro Tree Search (MCTS) or regular sampling-based inference. '
                'Sampling generates one single output image. '
                'MCTS iteratively refines outputs and sorts them in the "Compiled Image" tab based on their score. '
                'If you then click on an image preview its code is restored in the "TikZ Code" tab. '
                'Close the preview to again display the code stream of the current iteration.'
            )
            algorithm_radio = gr.Radio(
                choices=[(v, k) for k, v in ALGORITHMS.items()],
                value=algorithm,
                label="Inference algoritm",
                info=f"{lock_reason} {algorithm_info}" if lock else algorithm_info,
                interactive=not lock,
            )
            with gr.Accordion(label="Advanced"):
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=0.8,
                    label="Temperature",
                    info="The value used to modulate the next token probabilities.",
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.95,
                    label="Top-p",
                    info="If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top-p or higher are kept for generation.",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=5,
                    value=0,
                    label="Top-k",
                    info="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
                )
                exploration = gr.Slider(
                    minimum=0,
                    maximum=2,
                    step=0.1,
                    value=0.6,
                    visible=algorithm=="mcts",
                    label="Exploration coefficient",
                    info="Constant for MCTS to adjust the trade-off between exploration and exploitation.",
                )
                budget = gr.Slider(
                    minimum=0,
                    maximum=60,
                    step=1,
                    value=10,
                    visible=algorithm=="mcts",
                    label="Timeout",
                    info="The timeout in minutes after which MCTS should be stopped.",
                )
                strict = gr.Checkbox(
                    value=False,
                    label="Strict mode",
                    info="Treat recoverable errors same as fatal errors when computing scores for MCTS."
                )
                preprocess = gr.Checkbox(
                    value=True,
                    label="Preprocess image",
                    info="Trim whitespace and pad image to square."
                )

        mcts_programs, gallery_index = gr.State([]), gr.State(-1)
        # generate and compile TikZ
        generate_event = run_btn.click(
            check_inputs,
            inputs=[sketchpad],
            #show_progress="hidden",
            queue=False
        ).success(
            lambda: gr.Tabs(selected=0),
            outputs=tabs, # type: ignore
            show_progress="hidden",
            queue=False
        ).then(
            partial(inference, compile_timeout=timeout),
            # lots of inputs and outputs as we handle mcts and normal sampling with one function
            inputs=[base_model, sketchpad, temperature, top_p, top_k, exploration, budget, strict, preprocess, algorithm_radio],
            outputs=[stream_code, mcts_programs, result_image, result_gallery, tabs] # type: ignore
        )
        # cancel generation, when a stop button is pressed
        for btn in [clear_btn, stop_btn]:
            btn.click(fn=None, cancels=generate_event, show_progress="hidden", queue=False)
        # select either gallery or image output depending on inference algorithm and hide/unhide some components
        algorithm_radio.change(
            fn=lambda alg: (
                gr.Code("", visible=True),
                gr.Code("", visible=False),
                gr.Slider(visible=alg=="mcts"),
                gr.Slider(visible=alg=="mcts"),
                gr.Gallery(None, visible=alg=="mcts"),
                gr.Image(None, visible=alg=="sampling"),
                gr.Tabs(selected=0)
            ),
            inputs=algorithm_radio,
            outputs=[stream_code, gallery_code, budget, exploration, result_gallery, result_image, tabs],
            cancels=generate_event,
            show_progress="hidden",
            queue=False
        )
        # when an item is selected in the preview, show its code in the code tab
        def show_program(programs, e: gr.SelectData):
            return (e.index, gr.Code(visible=not e.selected), gr.Code(programs[e.index], visible=e.selected))
        result_gallery.select(
            fn=show_program,
            inputs=mcts_programs,
            outputs=[gallery_index, stream_code, gallery_code],
            show_progress="hidden",
            queue=False
        )
        # when the gallery is not focused, the above event handler is not
        # called even when another image is previewed. This handler works
        # around that issue.
        result_gallery.change(
            fn=lambda idx, programs: programs[idx] if 0 <= idx < len(programs) else "",
            inputs=[gallery_index, mcts_programs],
            outputs=gallery_code,
            show_progress="hidden",
            queue=False
        )
        # when the preview is closed restore the streaming code view
        preview_close_btn.click(
            fn=lambda: (gr.Code(visible=True), gr.Code(visible=False)),
            outputs = [stream_code, gallery_code],
            show_progress="hidden",
            queue=False
        )
        # scroll with output
        stream_code.change(
            fn=None,
            js="() => document.getElementById('stream-code').querySelector('.cm-gutters').scrollIntoView(false)",
            show_progress="hidden",
            queue=False
        )

        return demo
