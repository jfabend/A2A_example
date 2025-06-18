import logging
import os

import click
from dotenv import load_dotenv

from a2a_wrapper_database_agent import DatabaseAgent
from custom_types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
from push_notification_auth import PushNotificationSenderAuth
from server import A2AServer
from task_manager_database_agent import DatabaseAgentTaskManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8000)
def main(host, port):
    """Starts the Database Agent server."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
        skill = AgentSkill(
            id="database_access",
            name="Database",
            description="Counts articles stored in the database",
            tags=["count products", "inventory report"],
            examples=["How many different shoe articles do we have in our database?"],
        )
        agent_card = AgentCard(
            name="Database Agent",
            description="Can count articles in the database of the company",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DatabaseAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DatabaseAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=DatabaseAgentTaskManager(
                agent=DatabaseAgent(), notification_sender_auth=notification_sender_auth
            ),
            host=host,
            port=port,
        )

        server.app.add_route(
            "/.well-known/jwks.json",
            notification_sender_auth.handle_jwks_endpoint,
            methods=["GET"],
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
