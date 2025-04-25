# Dataset

We are working with a clinical dataset. To briefly summarize, any drug or treatment must undergo a series of verifications and tests before it can be approved for commercial use. Clinical trials represent the different testing phases that a treatment must go through. As illustrated in the figure below, several stages are required before a treatment can be validated.

![dummy1](./figs/ctrial_pres.png)

To conduct these trials—whether on a small or large scale—research laboratories need to recruit patients who meet specific medical and demographic conditions. The data we are using is derived from the eligibility criteria of various clinical trials. These criteria can be found on the following [website](https://clinicaltrials.gov/) and looks like this:

![dummy2](./figs/example_site.png)

We downloaded an annotated dataset (see the details in ``data/``) for **Named Entity Recognition**. The dataset is annotated in a *BIO-format*, meaning that we add an extra tag:

* **B**eginning: Assigned to the first word/token of an entity
* **I**inside: Assigned to all the words/tokens inside the entity (except the first word/token)
* **O**utside: Assigned to words/tokens that do not belong to an *specific* entity, so it depends on the domain and application. 

The BIO format offers clear boundaries for entities, marking the beginning (B) and continuation (I) of multi-word entities, while using "O" for non-entity words. It is flexible, simple to implement, and ensures easy extraction of entities. This format helps avoid ambiguity.

![dummy](./figs/example_1.png)
