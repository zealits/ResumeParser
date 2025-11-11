import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class GitHubProfileError(Exception):
    """Domain-specific exception for GitHub profile analysis failures."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class RepositoryDetails:
    name: str
    full_name: str
    description: Optional[str]
    url: str
    created_at: str
    updated_at: str
    pushed_at: str
    size: int
    stars: int
    forks: int
    watchers: int
    language: Optional[str]
    is_fork: bool
    is_archived: bool
    is_disabled: bool
    topics: List[str] = field(default_factory=list)
    languages: Dict[str, int] = field(default_factory=dict)


@dataclass
class RepositoriesSummary:
    total_repositories_analyzed: int
    total_user_repositories: int
    total_stars_analyzed: int
    total_forks_analyzed: int
    language_overview: Dict[str, float]
    primary_language: Optional[str]
    note: str


@dataclass
class ProfileInfo:
    username: str
    name: Optional[str]
    bio: Optional[str]
    company: Optional[str]
    location: Optional[str]
    email: Optional[str]
    blog: Optional[str]
    twitter_username: Optional[str]
    avatar_url: str
    profile_url: str
    followers: int
    following: int
    public_repos: int
    public_gists: int
    created_at: str
    updated_at: str


@dataclass
class AnalysisReport:
    profile_info: ProfileInfo
    repositories_summary: RepositoriesSummary
    repositories: List[RepositoryDetails]
    analysis: Dict[str, Any]
    api_usage: Dict[str, Any]


class GitHubProfileAnalyzer:
    """Service class responsible for fetching and analysing GitHub profile data."""

    def __init__(self, token: Optional[str] = None, timeout: float = 20.0) -> None:
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "StudyHQ-ResumeParser-GitHubAnalyzer",
        }
        self.token_provided = bool(token)
        if token:
            self.headers["Authorization"] = f"token {token}"
        self.timeout = timeout
        self.api_calls = 0

    # --------------------------------------------------------------------- #
    # GitHub API helpers
    # --------------------------------------------------------------------- #
    def _get(self, client: httpx.Client, url: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = client.get(url, params=params)
        self.api_calls += 1
        response.raise_for_status()
        return response.json()

    def extract_username(self, profile_url: str) -> Optional[str]:
        parsed = urlparse(profile_url)
        if not parsed.path:
            return None
        parts = [part for part in parsed.path.split("/") if part]
        return parts[0] if parts else None

    def get_user_info(self, client: httpx.Client, username: str) -> Dict[str, Any]:
        url = f"{self.base_url}/users/{username}"
        return self._get(client, url)

    def get_top_repos(self, client: httpx.Client, username: str, count: int) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/users/{username}/repos"
        params = {"per_page": count, "sort": "stars", "direction": "desc"}
        return self._get(client, url, params=params)

    def get_repo_languages(self, client: httpx.Client, repo_full_name: str) -> Dict[str, int]:
        url = f"{self.base_url}/repos/{repo_full_name}/languages"
        response = client.get(url)
        self.api_calls += 1

        if response.status_code == httpx.codes.FORBIDDEN:
            logger.warning("GitHub languages API rate limit reached for %s. Skipping language details.", repo_full_name)
            return {}

        response.raise_for_status()
        return response.json()

    # --------------------------------------------------------------------- #
    # Analysis logic
    # --------------------------------------------------------------------- #
    def analyze_profile(self, profile_url: str, repo_count: int = 10) -> AnalysisReport:
        if repo_count <= 0:
            raise GitHubProfileError("repo_count must be greater than zero", status_code=422)

        username = self.extract_username(profile_url)
        if not username:
            raise GitHubProfileError("Invalid GitHub profile URL. Could not determine username.", status_code=422)

        auth_context = "authenticated" if self.token_provided else "unauthenticated"
        logger.info(
            "Starting GitHub profile analysis for user %s (top %s repositories, %s requests).",
            username,
            repo_count,
            auth_context,
        )

        client = httpx.Client(headers=self.headers, timeout=self.timeout)
        try:
            try:
                user_info = self.get_user_info(client, username)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.NOT_FOUND:
                    raise GitHubProfileError(f"GitHub user '{username}' not found.", status_code=404) from exc
                raise GitHubProfileError(
                    f"Failed to fetch GitHub user info: {exc.response.text}",
                    status_code=exc.response.status_code,
                ) from exc

            try:
                repos = self.get_top_repos(client, username, repo_count)
            except httpx.HTTPStatusError as exc:
                raise GitHubProfileError(
                    f"Failed to fetch repositories: {exc.response.text}",
                    status_code=exc.response.status_code,
                ) from exc

            repos_data: List[RepositoryDetails] = []
            language_totals: Dict[str, int] = {}
            total_stars = 0
            total_forks = 0

            for repo in repos:
                repo_full_name = repo["full_name"]
                try:
                    languages = self.get_repo_languages(client, repo_full_name)
                except httpx.HTTPStatusError as exc:
                    logger.warning("Failed to fetch languages for %s: %s", repo_full_name, exc)
                    languages = {}

                for lang, bytes_count in languages.items():
                    language_totals[lang] = language_totals.get(lang, 0) + bytes_count

                repos_data.append(
                    RepositoryDetails(
                        name=repo["name"],
                        full_name=repo["full_name"],
                        description=repo.get("description"),
                        url=repo["html_url"],
                        created_at=repo["created_at"],
                        updated_at=repo["updated_at"],
                        pushed_at=repo["pushed_at"],
                        size=repo["size"],
                        stars=repo["stargazers_count"],
                        forks=repo["forks_count"],
                        watchers=repo["watchers_count"],
                        language=repo.get("language"),
                        is_fork=repo["fork"],
                        is_archived=repo["archived"],
                        is_disabled=repo["disabled"],
                        topics=repo.get("topics", []),
                        languages=languages,
                    )
                )

                total_stars += repo["stargazers_count"]
                total_forks += repo["forks_count"]

            language_overview: Dict[str, float] = {}
            primary_language: Optional[str] = None

            if language_totals:
                total_bytes = sum(language_totals.values())
                language_overview = {
                    lang: round(bytes_count / total_bytes * 100, 2)
                    for lang, bytes_count in language_totals.items()
                    if total_bytes > 0
                }
                language_overview = dict(sorted(language_overview.items(), key=lambda item: item[1], reverse=True))
                primary_language = max(language_totals.items(), key=lambda item: item[1])[0]

            profile_info = ProfileInfo(
                username=user_info["login"],
                name=user_info.get("name"),
                bio=user_info.get("bio"),
                company=user_info.get("company"),
                location=user_info.get("location"),
                email=user_info.get("email"),
                blog=user_info.get("blog"),
                twitter_username=user_info.get("twitter_username"),
                avatar_url=user_info["avatar_url"],
                profile_url=user_info["html_url"],
                followers=user_info["followers"],
                following=user_info["following"],
                public_repos=user_info["public_repos"],
                public_gists=user_info["public_gists"],
                created_at=user_info["created_at"],
                updated_at=user_info["updated_at"],
            )

            repositories_summary = RepositoriesSummary(
                total_repositories_analyzed=len(repos_data),
                total_user_repositories=user_info["public_repos"],
                total_stars_analyzed=total_stars,
                total_forks_analyzed=total_forks,
                language_overview=language_overview,
                primary_language=primary_language,
                note=f"Analysis based on top {len(repos_data)} repositories sorted by stars.",
            )

            analysis_section = self._build_analysis_section(repos_data)

            api_usage = {"total_api_calls": self.api_calls}

            logger.info("Completed GitHub analysis for %s. API calls used: %s", username, self.api_calls)

            return AnalysisReport(
                profile_info=profile_info,
                repositories_summary=repositories_summary,
                repositories=repos_data,
                analysis=analysis_section,
                api_usage=api_usage,
            )

        except httpx.RequestError as exc:
            raise GitHubProfileError(
                f"Network error while contacting GitHub: {exc}",
                status_code=502,
            ) from exc
        finally:
            client.close()

    @staticmethod
    def _build_analysis_section(repos_data: List[RepositoryDetails]) -> Dict[str, Any]:
        if not repos_data:
            return {
                "most_starred_repo": None,
                "most_forked_repo": None,
                "recently_updated": [],
                "oldest_repo_analyzed": None,
            }

        sorted_by_updated = sorted(repos_data, key=lambda repo: repo.updated_at, reverse=True)

        return {
            "most_starred_repo": max(repos_data, key=lambda repo: repo.stars),
            "most_forked_repo": max(repos_data, key=lambda repo: repo.forks),
            "recently_updated": sorted_by_updated[:5],
            "oldest_repo_analyzed": min(repos_data, key=lambda repo: repo.created_at),
        }


def analyze_github_profile(profile_url: str, repo_count: int = 10, token: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper that returns the analysis report as a plain dictionary."""
    analyzer = GitHubProfileAnalyzer(token=token)
    report = analyzer.analyze_profile(profile_url=profile_url, repo_count=repo_count)
    return asdict(report)

