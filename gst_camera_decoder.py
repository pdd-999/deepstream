import traceback
from collections import deque
from typing import Dict, Optional, Union

import gi
import numpy as np
import torch
from func_timeout import func_set_timeout
from loguru import logger
import time
from typing_extensions import Literal
from webrtc import MediasoupClient
from uuid import uuid4
from collections import deque
from utils import print_pad_templates_information

class SingleGSTCameraDecoder:
    """
    Decode each frame of an rtsp stream using GStreamer
    """
    def __init__(self,
            input_uri: str = "rtsp://192.168.40.5:8554/live.sdp2",
            width: int = 1920,
            height: int = 1080,
            fps: int = 30,
            use_gpu: bool = True,
            codec: Literal['h264', 'h265'] = 'h264',
            cam_id: str = None,
            frame_deque: deque = None,):

        self.input_uri = input_uri
        self.width = width
        self.height = height
        self.fps = fps
        self.use_gpu = use_gpu
        self.codec = codec
        self.cam_id = cam_id
        self.frame_deque = frame_deque

        global Gst
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        if self.use_gpu:
            global ghetto_nvds
            import ghetto_nvds
        self.temporary_deque = deque(maxlen=2)
        self._create_gstreamer_pipeline()

    # @func_set_timeout(3)
    def _create_gstreamer_pipeline(self):
        #NOTE: GPU only
        Gst.init(None)
        self.fake_sink_name = str(self.cam_id)
        self.frame_format, self.pixel_bytes = 'RGBA', 4

        self.pipeline = Gst.Pipeline.new(str(self.cam_id))

        self.rtspsrc = Gst.ElementFactory.make('rtspsrc', 'rtspsrc')
        if not self.rtspsrc:
            logger.error(f"Failed to create rtspsrc element")
        self.rtspsrc.set_property('location', self.input_uri)
        self.rtspsrc.set_property('latency', 0)
        self.rtspsrc.set_property('protocols', "GST_RTSP_LOWER_TRANS_TCP")
        
        def on_rtspsrc_pad_added(rtspsrc, pad, *user_data):
            print(f"Yooooo from on_rtspsrc_pad_added")
        self.rtspsrc.connect("pad-added", on_rtspsrc_pad_added)

        self.rtp_depay = Gst.ElementFactory.make(f'rtp{self.codec}depay', 'rtp-depay')
        if not self.rtp_depay:
            logger.error(f"Failed to create rtpdepay")

        self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
        if not self.decoder:
            logger.error(f"Failed to create nvv4l2decoder")

        self.converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
        if not self.converter:
            logger.error(f"Failed to create nvvideoconvert")
        self.converter.set_property("interpolation-method", 1)

        self.video_rate = Gst.ElementFactory.make("videorate", "video-rate")
        if not self.video_rate:
            logger.error(f"Failed to create videorate")

        video_rate_caps_string = Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1,format={self.frame_format}"
        )
        self.video_rate_caps = Gst.ElementFactory.make("capsfilter", "filter")
        if not self.video_rate_caps:
            logger.error(f"Failed to create capsfilter")

        self.video_rate_caps.set_property("caps", video_rate_caps_string)

        self.fakesink = Gst.ElementFactory.make("fakesink", str(self.cam_id))
        if not self.fakesink:
            logger.error(f"Failed to create fakesink")
        
        logger.info(f"Yooooooooooo")

        # Add element to pipeline
        self.pipeline.add(self.rtspsrc)
        self.pipeline.add(self.rtp_depay)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.converter)
        self.pipeline.add(self.video_rate)
        self.pipeline.add(self.video_rate_caps)
        self.pipeline.add(self.fakesink)

        # Link element
        self.rtspsrc.link(self.rtp_depay)
        self.rtp_depay.link(self.decoder)
        self.decoder.link(self.converter)
        self.converter.link(self.video_rate)
        self.video_rate.link(self.video_rate_caps)
        self.video_rate_caps.link(self.fakesink)

        # decodePipeline = f"""
        #     rtspsrc location="{self.input_uri}" latency=0 protocols="GST_RTSP_LOWER_TRANS_TCP" !
        #     rtp{self.codec}depay !
        #     nvv4l2decoder !
        # """

        # convertPipeline = f"""
        #     nvvideoconvert interpolation-method=1 ! 
        #     videorate !
        #     video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1,format={self.frame_format} !
        #     fakesink name={self.fake_sink_name}
        # """

        # # Init pipeline
        # try:
        #     self.pipeline = Gst.parse_launch(decodePipeline + convertPipeline)
        # except Exception as e:
        #     logger.error(f"######### ERROR WHEN INIT GST PIPELINE for camera {self.cam_id} ##########")
        #     logger.error(e)
        #     logger.error(f"######### ERROR WHEN INIT GST PIPELINE for camera {self.cam_id} ##########")

        self.bus = self.pipeline.get_bus()

        def on_frame_probe(pad, info):
            """
            Callback function for each frame
            """
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                logger.error(f"Camera {self.cam_id} unable to get GstBuffer")
                return Gst.PadProbeReturn.OK
            logger.info(f"Yoooo in on_frame_probe")
            # t0 = time.time()
            res = self.decode_frame_buffer_to_tensor(gst_buffer, pad.get_current_caps())
            # logger.debug(f"DECODE FRAME BUFFER TO TENSOR TOOK {(time.time() - t0)*1000}ms")
            
            try:
                frame = res.cpu().numpy()[...,[2,1,0]]
                self.frame_deque.appendleft(frame)
            except Exception as ex:
                logger.error(f"Camera {self.cam_id} failed to put frame into controller frame dequeue with: {ex}\n{traceback.format_exc()}")
                del res

            return Gst.PadProbeReturn.OK

        self.pipeline.get_by_name(self.fake_sink_name).get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            on_frame_probe,
        )

    def decode_frame_buffer_to_tensor(self, buf, caps) -> torch.Tensor:
        """
        Decode the frame buffer to gpu tensor
        :return: gpu tensor
        """
        # logger.info(f"Before mapping of {self.cam_id}")
        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        # logger.info(f"After mapping of {self.cam_id}")
        if is_mapped:
            try:
                logger.info(f"Yoooo in decode_frame_buffer_to_tensor of {self.cam_id}")
                # logger.info(f"Before ghetto nvds of {self.cam_id}")
                source_surface = ghetto_nvds.NvBufSurface(map_info)
                torch_surface = ghetto_nvds.NvBufSurface(map_info)
                # logger.info(f"After ghetto nvds of {self.cam_id}")

                dest_tensor = torch.zeros(
                    (torch_surface.surfaceList[0].height, torch_surface.surfaceList[0].width, 4),
                    dtype=torch.uint8,
                    device='cuda'
                )

                torch_surface.struct_copy_from(source_surface)
                assert(source_surface.numFilled == 1)
                assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

                # make torch_surface map to dest_tensor memory
                torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()

                # copy decoded GPU buffer (source_surface) into Pytorch tensor (torch_surface -> dest_tensor)
                torch_surface.mem_copy_from(source_surface)
            finally:
                buf.unmap(map_info)

            return dest_tensor[:, :, :3] # RGBA -> RGB

    def start(self):
        """
        Start the pipeline
        """
        self.pipeline.set_state(Gst.State.PLAYING)
        logger.info(f"Pipeline is started")

    def stop(self):
        """
        Stop the pipeline
        """
        self.pipeline.set_state(Gst.State.NULL)

if __name__=="__main__":
    frame_deque = deque(maxlen=10)
    
    cam_decoder = SingleGSTCameraDecoder(frame_deque=frame_deque, cam_id=666)
    cam_decoder.start()

    webrtc_client = MediasoupClient(
        mediasoup_server_uri=f'wss://192.168.40.4:4443/?roomId=4sapujyt&peerId={uuid4()}', 
        cam_id=0, 
        frame_deque=frame_deque
    )
