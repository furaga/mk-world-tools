from pathlib import Path

import cv2
import numpy as np
import obswebsocket


class OBSController:
    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        cache_dir: Path = Path(".cache"),
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.obs_ws_ = obswebsocket.obsws(host, port, password)
        self.obs_ws_.connect()
        self.cache_dir = cache_dir
        self.config = kwargs

    def set_browser_url(self, source_name: str, url: str):
        self.obs_ws_.call(
            obswebsocket.requests.SetSourceSettings(
                sourceName=source_name, sourceSettings={"url": url}
            )
        )

    def set_visible(self, source_name: str, visible: bool):
        self.obs_ws_.call(
            obswebsocket.requests.SetSceneItemProperties(
                item=source_name, visible=visible
            )
        )

    def get_text(self, source_name: str):
        return self.obs_ws_.call(
            obswebsocket.requests.GetSourceSettings(sourceName=source_name)
        ).datain["sourceSettings"]["text"]

    def set_text(self, source_name: str, new_text: str):
        self.obs_ws_.call(
            obswebsocket.requests.SetSourceSettings(
                sourceName=source_name, sourceSettings={"text": new_text}
            )
        )

    def set_color(self, source_name: str, new_rgb):
        r, g, b = new_rgb
        new_color = (b << 16) + (g << 8) + r
        self.obs_ws_.call(
            obswebsocket.requests.SetSourceSettings(
                sourceName=source_name, sourceSettings={"color": new_color}
            )
        )

    def capture_game_screen(
        self, sourceName: str = "映像キャプチャデバイス"
    ) -> np.ndarray:
        out_path = self.cache_dir / "obs_screenshot.jpg"
        self.obs_ws_.call(
            obswebsocket.requests.TakeSourceScreenshot(
                sourceName=sourceName,
                embedPictureFormat="jpg",
                saveToFilePath=str(out_path).replace("\\", "/"),
            )
        )
        img = cv2.imread(str(out_path))
        return img
