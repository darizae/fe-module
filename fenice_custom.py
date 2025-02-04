import torch
from typing import List, Dict
from metric.FENICE import FENICE


class FENICECustomClaims(FENICE):
    """
    Subclass of the original FENICE that injects custom claims
    instead of performing claim extraction.
    """

    def __init__(
            self,
            custom_claims_by_summary_id: Dict[str, List[str]],
            device: str,
            *args, **kwargs
    ):
        """
        :param custom_claims_by_summary_id: A dict that maps a summary-ID string to
                                            a list of claims (strings).
        Other arguments are the same as the parent FENICE class.
        """
        super().__init__(*args, **kwargs)
        # Store the custom claims in an internal variable
        self.custom_claims_by_summary_id = custom_claims_by_summary_id
        self.device = device

    def cache_claims(self, summaries: List[str]):
        """
        Override the default claim-extraction pipeline to inject our own claims.
        """
        for idx, summary_text in enumerate(summaries):
            summary_id = self.get_id(idx, summary_text)
            # Retrieve the claims for this summary from the dictionary
            if summary_id in self.custom_claims_by_summary_id:
                self.claims_cache[summary_id] = self.custom_claims_by_summary_id[summary_id]
            else:
                raise ValueError("No system claim for summary id {}".format(summary_id))
