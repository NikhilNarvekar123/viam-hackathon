import asyncio
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, call

from tools.gripper_demo import cycle

ROOT = Path(__file__).resolve().parents[1]


def test_cycle_order():
    gripper = AsyncMock()
    asyncio.run(cycle(gripper, 0))
    assert gripper.mock_calls == [call.open(), call.grab(), call.open()]


def test_real_requires_confirmation():
    result = subprocess.run([sys.executable, str(ROOT / "tools/gripper_demo.py"), "--real"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--confirm-motion" in result.stderr


def test_default_fake():
    result = subprocess.run([sys.executable, str(ROOT / "tools/gripper_demo.py"), "--delay", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "FAKE MODE" in result.stdout
    assert "Cycle complete" in result.stdout
