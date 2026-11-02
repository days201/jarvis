"""Check real cloud vision and tools with a generated image and simulated motors."""

import json
import sys
import threading

from PIL import Image, ImageDraw, ImageFont

from assistant import AssistantError, DeepSeekClient
from robot import Robot
from test_robot import SimulatedCamera


class SimulatedRobot(Robot):
    def context(self):
        return {**super().context(), "simulation": True, "note": "Motors and image source are simulated for this test."}


def main():
    login = json.loads(sys.stdin.read())
    client = DeepSeekClient(login.pop("api_key"))
    del login
    camera = SimulatedCamera()
    robot = SimulatedRobot(camera, (0, 0), start=False)
    robot.following = False
    image = Image.new("RGB", (800, 450), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 70, 330, 370), fill="blue")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 110)
    draw.text((430, 140), "42", font=font, fill="black")
    done = threading.Event()
    def frames():
        while not done.is_set():
            robot.frames.push(image)
            done.wait(0.13)
    threading.Thread(target=frames, daemon=True).start()
    try:
        answer = "".join(client.reply(
            "This is a simulation test. Use the look tool to turn right by exactly 10 degrees once. Then tell me the color of the rectangle and the printed number in the image in one short sentence.",
            threading.Event(), robot=robot, new_conversation=True))
        success = (len(camera.moves) == 1 and abs(camera.moves[0][0] - 10 / 170) < 0.0001
                   and "blue" in answer.lower() and "42" in answer)
        print(json.dumps({"passed": success, "simulation": True, "reply": answer,
                          "simulated_motor_moves": camera.moves, "metrics": client.metrics,
                          "images_in_saved_history": "data:image" in json.dumps(client._history)}), flush=True)
        return 0 if success else 1
    except AssistantError as error:
        print(json.dumps({"passed": False, "error": str(error)}), flush=True)
        return 1
    finally:
        done.set()
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
