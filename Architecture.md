
# Sector Architecture

![Sector Architecture](img/sector.jpg)

**Sector** is designed as a flexible and extensible framework for semantic extraction and comparison, allowing users to customize and adapt its components to fit various use cases. Below is an overview of how each component contributes to Sector's highly customizable and modular design.

## Architecture Components

1. **Pre-processing and Embedding Functions**: Sector allows users to use pre-defined or bring in custom pre-processing and embedding functions. These functions enable tailored pre-processing and representation of both the context and LLM responses, improving relevance and accuracy based on specific needs.

2. **Extractor and Matcher Flexibility**: The **Extractor** component extracts relevant sections from the context, while the **Matcher** performs either semantic or traditional matching. Users can adjust the **Search Space** parameters—such as `Max Window Size`, `Search Algorithm`, and `Early Stopping Threshold`—to control the scope and depth of the context search, balancing speed and accuracy as needed.

3. **Custom and Pre-defined Comparators**: Sector supports both **Pre-defined Comparators** and **Custom Comparators**, enabling users to use out of the box metrics or standard similarity metrics (like BLEU, GLEU, and METEOR) or to bring their own comparison functions. This versatility allows Sector to meet diverse evaluation strategies and domain-specific requirements.

4. **Output and Scoring**: The output includes the best-matched context and confidence scores, which can be fine-tuned using custom comparator settings. This helps users assess the quality of LLM responses according to their specific objectives.

## Extensibility and Customization

The Sector architecture is built with a **modular, plug-and-play design**, allowing users to seamlessly integrate and customize individual components or utilize the entire framework based on their requirements. Each component—**Pre-processing**, **Embedding**, **Extractor**, **Matcher**, and **Comparator**—can be individually configured or replaced with custom functions. Users have the flexibility to employ Sector’s predefined tools or substitute them with their own, tailoring the workflow to fit specific use cases.

For instance, you can choose to only use the **Extractor** and **Matcher** for relevance filtering, or go further by integrating custom **Comparators** for specialized evaluation metrics. The **Search Space** parameters (such as `Max Window Size`, `Search Algorithm`, and `Early Stopping Threshold`) provide additional customization, allowing fine-tuning for either comprehensive analysis or optimized performance. This plug-and-play flexibility makes Sector exceptionally adaptable, enabling it to meet diverse needs—whether as a complete solution or as individual components within a larger system.

---

This modularity and extensibility make Sector a powerful tool for applications requiring nuanced, context-aware evaluation, whether for real-time processing or thorough batch analysis.
