import logging
from typing import List
from ..nodes.iot_client import IoTClient
from ..nodes.fog_node import FogNode
from ..nodes.cloud_server import CloudServer

logger = logging.getLogger(__name__)

class SplitFederatedProtocol:

    def __init__(self, clients: List[IoTClient], fogs: List[FogNode], cloud: CloudServer):
        self.clients = clients
        self.fogs = fogs
        self.cloud = cloud

    def run_round(self):

        logger.info("Starting IoT forward passes")
        packets = []
        for client in self.clients:
            smashed, labels = client.forward_pass()
            packets.append((client, smashed, labels))

        logger.info("Fog processing")
        processed = []
        for (client, smashed, labels) in packets:
            fog = self.fogs[0]
            fog_out = fog.forward_pass(smashed)
            processed.append((client, fog_out, labels))

        logger.info("Cloud inference and optimization")
        for (client, fog_out, labels) in processed:
            loss, _ = self.cloud.infer_and_loss(fog_out.detach(), labels)
            self.cloud.step(loss)

        logger.info("Round complete")
