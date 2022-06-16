import asyncio
import json
import secrets
import ssl
import sys
import threading
import traceback
from asyncio.futures import Future
from collections import deque
from random import random
from typing import Any, Awaitable, Dict, Optional, TypeVar
from uuid import uuid4

import cv2
import numpy as np
import websockets
from aiortc import VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame
from loguru import logger
from pymediasoup import AiortcHandler, Device
from pymediasoup.consumer import Consumer
from pymediasoup.data_consumer import DataConsumer
from pymediasoup.data_producer import DataProducer
from pymediasoup.producer import Producer
from pymediasoup.sctp_parameters import SctpStreamParameters
from pymediasoup.transport import Transport


class DeQueueVideoStreamTrack(VideoStreamTrack):
    """
    A video track that returns frames from a deque.
    """
    def __init__(self, deque: deque):
        super().__init__()
        self.deque = deque
        if self.deque.maxlen is None:
            logger.warning("deque for WebRTC stream should have a maxlen arround 5 frames, \
                don't have maxlen set might cause memory leak when there are no clients connected to consume the stream")

    def _check_rgb(self, frame: np.ndarray) -> np.ndarray:
        # NOTE: So we can simply force RGB conversion right here using
        # cv2.cvtColor, I did it before and turn out in some weird cases
        # cvtColor will cause 100% CPU usage, so we turn to raise Exception
        # when the frame are not RGB insted of using cv2.cvtColor.
        if len(frame.shape) == 3 and frame.shape[-1] == 3:
            return
        raise ValueError("Frame is not RGB")

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        while True:
            try:
                frame = self.deque.pop()
            except IndexError:
                await asyncio.sleep(1/100)
                continue
            try:
                self._check_rgb(frame)
                frame = VideoFrame.from_ndarray(frame, format="bgr24")
                frame.pts = pts
                frame.time_base = time_base
                return frame
            except Exception as ex:
                logger.error(f"Fail to create VideoFrame with error: {ex}")
                logger.error(traceback.format_exc())
                continue

T = TypeVar("T")

def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception in asyncio: {msg}")

class MediasoupClient:
    def __init__(self, 
                mediasoup_server_uri: str,
                frame_deque: DeQueueVideoStreamTrack,
                cam_id: int=0):
        self.cam_id = cam_id

        self._uri = mediasoup_server_uri
        self._recorder = MediaBlackhole()
        # Save answers temporarily
        self._answers: Dict[str, Future] = {}
        self._websocket = None
        self._device = None

        self._tracks = []
        if not isinstance(frame_deque, DeQueueVideoStreamTrack):
            frame_deque = DeQueueVideoStreamTrack(frame_deque)
            
        self.frame_deque = frame_deque
        self._tracks.append(frame_deque)

        self._sendTransport: Optional[Transport] = None
        self._recvTransport: Optional[Transport] = None

        self._producers = []
        self._consumers = []
        self._tasks = []
        self._closed = False

        self.main_thread = threading.Thread(target=self.asyncio_thread, args=())
        self.main_thread.start()

    def asyncio_thread(self):
        self._loop = asyncio.new_event_loop()
        self._loop.set_exception_handler(handle_exception)

        try:
            self._loop.run_until_complete(
                self.run()
            )
        except:
            logger.warning(f"Error in asyncio_thread: {traceback.format_exc()}")

    # websocket receive task
    async def recv_msg_task(self):
        while True:
            await asyncio.sleep(0.5)
            if self._websocket != None:
                message = json.loads(await self._websocket.recv())
                if message.get('response'):
                    if message.get('id') != None:
                        self._answers[message.get('id')].set_result(message)
                elif message.get('request'):
                    if message.get('method') == 'newConsumer':
                        await self.consume(
                            id=message['data']['id'],
                            producerId=message['data']['producerId'],
                            kind=message['data']['kind'],
                            rtpParameters=message['data']['rtpParameters']
                        )
                        response = {
                            'response': True,
                            'id': message['id'],
                            'ok': True,
                            'data': {}
                        }
                        await self._websocket.send(json.dumps(response))
                    elif message.get('method') == 'newDataConsumer':
                        await self.consumeData(
                            id=message['data']['id'],
                            dataProducerId=message['data']['dataProducerId'],
                            label=message['data']['label'],
                            protocol=message['data']['protocol'],
                            sctpStreamParameters=message['data']['sctpStreamParameters']
                        )
                        response = {
                            'response': True,
                            'id': message['data']['id'],
                            'ok': True,
                            'data': {}
                        }
                        await self._websocket.send(json.dumps(response))
                elif message.get('notification'):
                    # print(message)
                    pass

    # wait for answer ready        
    async def _wait_for(
        self, fut: Awaitable[T], timeout: Optional[float], **kwargs: Any
    ) -> T:
        try:
            return await asyncio.wait_for(fut, timeout=timeout, **kwargs)
        except asyncio.TimeoutError:
            raise Exception("Operation timed out")

    async def _send_request(self, request):
        self._answers[request['id']] = self._loop.create_future()
        await self._websocket.send(json.dumps(request))

    # Generates a random positive integer.
    def generateRandomNumber(self) -> int:
        return round(random() * 10000000)

    async def run(self):
        # self.ssl_context = ssl._create_unverified_context()
        # self._websocket = await websockets.connect(self._uri, subprotocols=['protoo'], ssl=self.ssl_context)

        self._websocket = await websockets.connect(self._uri, subprotocols=['protoo'])

        if sys.version_info < (3, 7):
            task_run_recv_msg = asyncio.ensure_future(self.recv_msg_task())
        else:
            task_run_recv_msg = asyncio.create_task(self.recv_msg_task())
        self._tasks.append(task_run_recv_msg)

        await self.load()
        await self.createSendTransport()
        await self.createRecvTransport()
        await self.produce()

        await task_run_recv_msg
        
    async def load(self):
        # Init device
        self._device = Device(handlerFactory=AiortcHandler.createFactory(tracks=self._tracks))

        # Get Router RtpCapabilities
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'getRouterRtpCapabilities',
            'data': {}
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)

        # Load Router RtpCapabilities
        await self._device.load(ans['data'])
    
    async def createSendTransport(self):
        if self._sendTransport != None:
            return
        # Send create sendTransport request
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'createWebRtcTransport',
            'data': {
                'forceTcp': False,
                'producing': True,
                'consuming': False,
                'sctpCapabilities': self._device.sctpCapabilities.dict()
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)

        # Create sendTransport
        self._sendTransport = self._device.createSendTransport(
            id=ans['data']['id'], 
            iceParameters=ans['data']['iceParameters'], 
            iceCandidates=ans['data']['iceCandidates'], 
            dtlsParameters=ans['data']['dtlsParameters'],
            sctpParameters=ans['data']['sctpParameters']
        )

        @self._sendTransport.on('connect')
        async def on_connect(dtlsParameters):
            reqId = self.generateRandomNumber()
            req = {
                "request":True,
                "id":reqId,
                "method":"connectWebRtcTransport",
                "data":{
                    "transportId": self._sendTransport.id,
                    "dtlsParameters": dtlsParameters.dict(exclude_none=True)
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
        
        @self._sendTransport.on('produce')
        async def on_produce(kind: str, rtpParameters, appData: dict):
            reqId = self.generateRandomNumber()
            req = {
                "id": reqId,
                'method': 'produce',
                'request': True,
                'data': {
                    'transportId': self._sendTransport.id,
                    'kind': kind,
                    'rtpParameters': rtpParameters.dict(exclude_none=True),
                    'appData': appData
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
            return ans['data']['id']

        @self._sendTransport.on('producedata')
        async def on_producedata(
            sctpStreamParameters: SctpStreamParameters,
            label: str,
            protocol: str,
            appData: dict
        ):
            reqId = self.generateRandomNumber()
            req = {
                "id": reqId,
                'method': 'produceData',
                'request': True,
                'data': {
                    'transportId': self._sendTransport.id,
                    'label': label,
                    'protocol': protocol,
                    'sctpStreamParameters': sctpStreamParameters.dict(exclude_none=True),
                    'appData': appData
                }
            }
            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
            return ans['data']['id']

    async def produce(self):
        if self._sendTransport == None:
            await self.createSendTransport()

        # Join room
        reqId = self.generateRandomNumber()
        req = {
            "request":True,
            "id":reqId,
            "method":"join",
            "data":{
                "displayName": f"{self.cam_id}",
                "device":{
                    "flag":"python",
                    "name":"python","version":"0.1.0"
                },
                "rtpCapabilities": self._device.rtpCapabilities.dict(exclude_none=True),
                "sctpCapabilities": self._device.sctpCapabilities.dict(exclude_none=True)
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)
        # produce
        videoProducer: Producer = await self._sendTransport.produce(
            track=self.frame_deque,
            stopTracks=False,
            appData={}
        )
        self._producers.append(videoProducer)

        # produce data
        await self.produceData()
    
    async def produceData(self):
        if self._sendTransport == None:
            await self.createSendTransport()

        dataProducer: DataProducer = await self._sendTransport.produceData(
            ordered=False,
            maxPacketLifeTime=5555,
            label='chat',
            protocol='',
            appData={'info': "my-chat-DataProducer"}
        )
        self._producers.append(dataProducer)
        while True:
            await asyncio.sleep(1)
            # dataProducer.send('hello')
    
    async def createRecvTransport(self):
        if self._recvTransport != None:
            return
        # Send create recvTransport request
        reqId = self.generateRandomNumber()
        req = {
            'request': True,
            'id': reqId,
            'method': 'createWebRtcTransport',
            'data': {
                'forceTcp': False,
                'producing': False,
                'consuming': True,
                'sctpCapabilities': self._device.sctpCapabilities.dict()
            }
        }
        await self._send_request(req)
        ans = await self._wait_for(self._answers[reqId], timeout=15)

        # Create recvTransport
        self._recvTransport = self._device.createRecvTransport(
            id=ans['data']['id'], 
            iceParameters=ans['data']['iceParameters'], 
            iceCandidates=ans['data']['iceCandidates'], 
            dtlsParameters=ans['data']['dtlsParameters'],
            sctpParameters=ans['data']['sctpParameters']
        )

        @self._recvTransport.on('connect')
        async def on_connect(dtlsParameters):
            reqId = self.generateRandomNumber()
            req = {
                "request":True,
                "id":reqId,
                "method":"connectWebRtcTransport",
                "data":{
                    "transportId": self._recvTransport.id,
                    "dtlsParameters": dtlsParameters.dict(exclude_none=True)
                }
            }

            await self._send_request(req)
            ans = await self._wait_for(self._answers[reqId], timeout=15)
        
    async def consume(self, id, producerId, kind, rtpParameters):
        if self._recvTransport == None:
            await self.createRecvTransport()
        consumer: Consumer = await self._recvTransport.consume(
            id=id,
            producerId=producerId,
            kind=kind,
            rtpParameters=rtpParameters
        )
        self._consumers.append(consumer)
        self._recorder.addTrack(consumer.track)
        await self._recorder.start()

    async def consumeData(self, id, dataProducerId, sctpStreamParameters, label=None, protocol=None, appData={}):
        pass
        dataConsumer: DataConsumer = await self._recvTransport.consumeData(
            id=id,
            dataProducerId=dataProducerId,
            sctpStreamParameters=sctpStreamParameters,
            label=label,
            protocol=protocol,
            appData=appData
        )
        self._consumers.append(dataConsumer)
        @dataConsumer.on('message')
        def on_message(message):
            print(f'DataChannel {label}-{protocol}: {message}')

    async def close(self):
        try:
            for consumer in self._consumers:
                await consumer.close()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")
        
        try:
            for producer in self._producers:
                await producer.close()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")

        try:
            for task in self._tasks:
                task.cancel()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")

        try:
            if self._sendTransport:
                await self._sendTransport.close()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")

        try:
            if self._recvTransport:
                await self._recvTransport.close()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")

        try:
            await self._recorder.stop()
        except:
            logger.warning(f"Exception when closing Mediasoup: {traceback.format_exc()}")

    def stop(self):
        try:
            with threading.Lock():
                asyncio.set_event_loop(self._loop)
                for task in asyncio.Task.all_tasks():
                    task.cancel()

            if self.main_thread.is_alive():
                self.main_thread.join()

            self._loop.run_until_complete(self._websocket.close())
            self._loop.run_until_complete(self.close())
            self._loop.close()
            logger.info(f"Done stopping Mediasoup")
        except:
            logger.warning(f"Exception when stopping Mediasoup: {traceback.format_exc()}")
