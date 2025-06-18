import logging
import os

import click
from dotenv import load_dotenv

from a2a_wrapper_currency_agent import CurrencyAgent
from custom_types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
from push_notification_auth import PushNotificationSenderAuth
from server import A2AServer
from task_manager import AgentTaskManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8001)
def main(host, port):
    '''Starts the Currency Agent server.'''
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise MissingAPIKeyError('OPENAI_API_KEY environment variable not set.')

        capabilities = AgentCapabilities(streaming=False, pushNotifications=False)

        skill = AgentSkill(
            id='currency_exchange',
            name='Currency Exchange',
            description='Converts amounts between USD and EUR',
            tags=['currency', 'exchange', 'conversion'],
            examples=['Convert 100 USD to EUR', 'How much is 50 EUR in USD?'],
        )

        agent_card = AgentCard(
            name='Currency Agent',
            description='Performs USD/EUR currency conversions using a built‑in calculator.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=['text/plain'],
            defaultOutputModes=['text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=CurrencyAgent(), notification_sender_auth=notification_sender_auth
            ),
            host=host,
            port=port,
        )

        server.app.add_route(
            '/.well-known/jwks.json',
            notification_sender_auth.handle_jwks_endpoint,
            methods=['GET'],
        )

        logger.info(f'Starting server on {host}:{port}')
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
