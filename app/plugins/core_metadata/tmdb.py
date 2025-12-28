import os
from typing import Any

from curl_cffi.requests import AsyncSession, RequestsError
from pydantic import BaseModel, Field

from app.core.context import MMCPContext
from app.core.plugin_interface import MMCPTool


# 1. Define Input Schema
class TMDbMetadataInput(BaseModel):
    title: str = Field(..., description="The movie or TV show title")
    year: int | None = Field(None, description="Release year (optional but recommended)")
    type: str = Field("movie", description="Type of media: 'movie' or 'tv'")


# 2. Define Logic
class TMDbLookupTool(MMCPTool):
    """
    TMDb Metadata Lookup Tool.

    This product uses the TMDb API but is not endorsed or certified by TMDb.
    See: https://www.themoviedb.org/api-terms-of-use
    """

    @property
    def name(self) -> str:
        return "tmdb_lookup_metadata"

    @property
    def status_description(self) -> str:
        return "Movie/TV metadata lookup via TMDb API"

    @property
    def description(self) -> str:
        return "Finds movie/TV show metadata (ID, year, overview, poster) using The Movie Database (TMDb) API."

    @property
    def input_schema(self) -> type[BaseModel]:
        return TMDbMetadataInput

    def is_available(self) -> bool:
        """
        Check if TMDb API key is configured.

        Note: This method doesn't have access to context, so it checks env directly.
        The execute() method uses context-injected config for actual execution.
        """
        import os

        return bool(os.getenv("TMDB_API_KEY"))

    def get_status_info(self) -> dict[str, Any]:
        """
        Return TMDb plugin status information.

        Plugin manages its own status - no core knowledge needed.
        """
        import os

        api_key = os.getenv("TMDB_API_KEY")

        return {
            "service_name": "TMDb API",
            "configured": bool(api_key),
            "description": self.status_description,
        }

    def _extract_year(self, date_str: str | None) -> int | None:
        """Extract year from TMDb date string (YYYY-MM-DD format)."""
        if not date_str:
            return None
        try:
            # TMDb dates are in YYYY-MM-DD format
            return int(date_str.split("-")[0])
        except (ValueError, AttributeError):
            return None

    async def execute(
        self, context: MMCPContext, title: str, year: int | None = None, type: str = "movie"
    ) -> dict[str, Any]:
        """
        Queries TMDb API v3 for content metadata.

        Uses modern Bearer token authentication (API Read Access Token).
        Context injection ensures tools never fetch config themselves - all config
        comes from MMCPContext.static.config for security and testability.
        """
        # Inject plugin config into context if not already present
        if "TMDB_API_KEY" not in context.static.config:
            api_key_value = os.getenv("TMDB_API_KEY")
            context.inject_plugin_config("TMDB_API_KEY", api_key_value)

        # Use context-injected secrets/config instead of os.getenv
        # This allows us to mock configs during testing or change providers easily
        api_key = context.static.config.get("TMDB_API_KEY")
        if not api_key:
            return {"error": "TMDB_API_KEY is not configured in the system context."}

        # TMDb API v3 endpoint: /search/movie or /search/tv
        url = f"https://api.themoviedb.org/3/search/{type}"

        params = {
            "query": title,
            "page": 1,
        }
        if year:
            # TMDb uses 'year' for movies, 'first_air_date_year' for TV shows
            key = "year" if type == "movie" else "first_air_date_year"
            params[key] = year

        # Standard Bearer Token Authentication - modern and secure
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        # Using curl_cffi AsyncSession for native non-blocking I/O
        # This is more efficient than asyncio.to_thread() and doesn't spawn OS threads
        try:
            async with AsyncSession(impersonate="chrome") as session:
                response = await session.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            if not results:
                return {"error": f"No results found for '{title}'"}

            # Return the top match with normalized fields
            top_match = results[0]
            date_field = "release_date" if type == "movie" else "first_air_date"
            date_str = top_match.get(date_field)

            return {
                "title": top_match.get("title") or top_match.get("name"),
                "year": self._extract_year(date_str),
                "release_date": date_str,  # Full date string for reference
                "overview": top_match.get("overview"),
                "tmdb_id": top_match.get("id"),
                "poster_path": top_match.get("poster_path"),  # Can construct full URL if needed
                "type": type,
            }

        except RequestsError as e:
            # Handle 401 authentication errors specifically
            if "401" in str(e):
                return {
                    "error": "TMDB Authentication failed. Check if you used the 'API Read Access Token'."
                }
            return {"error": f"Network error connecting to TMDb: {str(e)}"}
        except Exception as e:
            return {"error": f"Metadata lookup failed: {str(e)}"}
