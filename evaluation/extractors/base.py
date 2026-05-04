from abc import ABC, abstractmethod

from evaluation.types import EvalSample


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, datasets: list[str]) -> dict[str, list[EvalSample]]:
        """Return {query_type: [EvalSample]} for each requested dataset.

        Traces with query_type not in datasets are skipped. Query types with
        no matching traces return an empty list rather than raising an error.
        reference is always None — golden attachment is the runner's responsibility.
        """
