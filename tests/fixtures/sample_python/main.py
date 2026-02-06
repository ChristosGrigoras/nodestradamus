"""Sample main module for testing Python dependency analysis."""

from services.user import UserService
from utils import helper_function


def main():
    """Main entry point."""
    helper_function()
    service = UserService()
    service.get_user("1")


if __name__ == "__main__":
    main()
