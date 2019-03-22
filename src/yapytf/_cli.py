import click
import click_log
import logging


logger = logging.getLogger(__name__)
click_log.basic_config(logger)


@click.group()
@click_log.simple_verbosity_option(logger)
@click.version_option()
def main(**opts):
    """
    Coming soon
    """
