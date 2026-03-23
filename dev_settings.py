# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
#!/usr/bin/env python3
import logging
import os

logger: logging.Logger = logging.getLogger(__name__)

# DEV_MODE is set to False by default.
DEV_MODE: bool = os.getenv("FASTMOE_DEV_MODE", "0") == "1"
logger.info(f"FastMOE DEV_MODE is set to {DEV_MODE}.")

VERBOSE_LEVEL: int = 0


def set_dev_mode(val: bool) -> None:
    global DEV_MODE
    if os.getenv("FASTMOE_DEV_MODE") is None:
        DEV_MODE = val
        logger.info(f"FastMOE DEV_MODE is set to {DEV_MODE}.")
    else:
        logger.warning("set_dev_mode() is ignored when FASTMOE_DEV_MODE is set.")


def is_dev_mode() -> bool:
    global DEV_MODE
    return DEV_MODE


def set_verbose_level(level: int) -> None:
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = level


def get_verbose_level() -> int:
    global VERBOSE_LEVEL
    return VERBOSE_LEVEL
