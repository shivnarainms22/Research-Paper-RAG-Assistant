"""
Prompt templates for the Research Paper RAG system.
"""


class PromptTemplates:
    """
    Collection of prompt templates for different query types.
    """

    SYSTEM_PROMPT = """You are an expert AI research assistant specializing in machine learning and artificial intelligence papers. Your role is to:

1. Answer technical questions accurately based on the provided research paper excerpts
2. Explain complex mathematical concepts and equations in plain English
3. Compare and contrast different approaches across papers
4. Always cite your sources with the exact paper title and page numbers
5. Acknowledge when information is not available in the provided context

When explaining equations:
- Break down each component and its meaning
- Provide intuitive explanations using analogies when helpful
- Explain the significance of the equation in the broader context

FORMATTING GUIDELINES:
- Use **bold** for key terms and important concepts
- Use headers (###) to organize long responses into sections
- Use bullet points for lists of items
- Use numbered lists for sequential steps or ranked items
- Keep paragraphs concise and focused
- Use `inline code` for technical terms, variables, or formulas when appropriate

Always be precise about what comes from the papers versus your general knowledge. Mark general knowledge explanations clearly.

Provide comprehensive, well-structured responses that directly address the user's question."""

    QA_TEMPLATE = """Based on the following excerpts from research papers, please answer the user's question.

RESEARCH PAPER CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. Cite specific papers and page numbers when referencing content
3. If the context doesn't contain enough information, say so clearly
4. For technical concepts, provide both the formal definition and a plain English explanation
5. Use clear formatting with headers, bullet points, and paragraphs for readability
6. Structure your response logically based on what the user is asking
7. Be comprehensive but concise - include all relevant information without unnecessary verbosity

ANSWER:"""

    EXPLAIN_MATH_TEMPLATE = """Based on the following research paper excerpts containing mathematical content, please explain the equations and mathematical concepts.

RESEARCH PAPER CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Identify all relevant equations in the context
2. Break down each equation component by component
3. Explain what each symbol/variable represents
4. Provide an intuitive, plain-English explanation
5. Explain why this mathematical formulation is used
6. Give a concrete example if possible

EXPLANATION:"""

    COMPARE_PAPERS_TEMPLATE = """Based on the following excerpts from multiple research papers, please compare and contrast them.

PAPER EXCERPTS:
{context}

COMPARISON QUERY:
{question}

INSTRUCTIONS:
1. Identify the key aspects relevant to the comparison
2. Create a structured comparison addressing:
   - Similarities
   - Key differences
   - Relative strengths and weaknesses
   - Use cases where each approach excels
3. Cite specific papers when making claims
4. Summarize with a clear recommendation or conclusion if applicable

COMPARISON:"""

    SUMMARIZE_SECTION_TEMPLATE = """Based on the following excerpts from a research paper, provide a comprehensive explanation.

PAPER CONTENT:
{context}

USER REQUEST:
{question}

INSTRUCTIONS:
1. Start with the paper's main objective and motivation
2. Explain the key methodology or approach
3. Describe the main contributions and findings
4. Explain technical concepts in accessible language
5. Note any important results or experiments
6. Mention limitations if relevant
7. Use clear formatting:
   - Use **bold** for key terms and concepts
   - Use bullet points for lists
   - Use headers (###) to organize sections
   - Keep paragraphs focused and readable

RESPONSE:"""

    LIMITATIONS_TEMPLATE = """Based on the following research paper excerpts, identify and explain the limitations.

RESEARCH PAPER CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Identify explicitly stated limitations from the paper
2. Note any implicit limitations you can infer
3. Consider:
   - Methodological limitations
   - Dataset/experimental limitations
   - Scope limitations
   - Assumptions that may not hold
4. Cite specific sections when referencing claims

LIMITATIONS ANALYSIS:"""

    @classmethod
    def get_template(cls, query_type: str) -> str:
        """
        Get the appropriate template for a query type.
        
        Args:
            query_type: One of 'qa', 'math', 'compare', 'summarize', 'limitations'
            
        Returns:
            The template string.
        """
        templates = {
            "qa": cls.QA_TEMPLATE,
            "math": cls.EXPLAIN_MATH_TEMPLATE,
            "compare": cls.COMPARE_PAPERS_TEMPLATE,
            "summarize": cls.SUMMARIZE_SECTION_TEMPLATE,
            "limitations": cls.LIMITATIONS_TEMPLATE,
        }

        return templates.get(query_type, cls.QA_TEMPLATE)

    @classmethod
    def detect_query_type(cls, query: str) -> str:
        """
        Automatically detect the query type from the question.
        
        Args:
            query: The user's question.
            
        Returns:
            Query type string.
        """
        query_lower = query.lower()

        # Check for math/equation queries
        math_indicators = ["equation", "formula", "math", "derive", "∇", "∑", "integral", 
                          "loss function", "objective function", "gradient", "optimization"]
        if any(ind in query_lower for ind in math_indicators):
            return "math"

        # Check for comparison queries
        compare_indicators = ["compare", "difference", "different", "versus", "vs", "between",
                             "how does .* differ", "contrast", "similarities"]
        if any(ind in query_lower for ind in compare_indicators):
            return "compare"

        # Check for summarization queries
        summarize_indicators = ["summarize", "summary", "overview", "explain briefly", "main points",
                               "what is this paper about", "explain this paper", "in detail",
                               "tell me about", "describe the paper", "key ideas", "main ideas",
                               "what does this paper", "contribution"]
        if any(ind in query_lower for ind in summarize_indicators):
            return "summarize"

        # Check for limitations queries
        limitations_indicators = ["limitation", "weakness", "drawback", "problem", "issue", "fail",
                                 "shortcoming", "disadvantage", "challenge", "criticism"]
        if any(ind in query_lower for ind in limitations_indicators):
            return "limitations"

        # Default to general QA
        return "qa"

