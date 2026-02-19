"""
Lightning Web Component (LWC) processor for Salesforce ingestion.

Extracts component metadata, decorators, imports, and template bindings
from LWC JavaScript and HTML files.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.shared.patterns.processor import IProcessor, ExtractedContent


@dataclass
class LWCProperty:
    """Represents a decorated property in an LWC component."""

    name: str
    decorator: str  # @api, @track, @wire
    wire_adapter: Optional[str] = None  # For @wire: the adapter name
    wire_params: Optional[Dict[str, str]] = None  # For @wire: configuration
    default_value: Optional[str] = None
    is_getter: bool = False
    is_setter: bool = False
    docstring: Optional[str] = None


@dataclass
class LWCMethod:
    """Represents a method in an LWC component."""

    name: str
    parameters: List[str] = field(default_factory=list)
    is_handler: bool = False  # Event handler (handleX naming convention)
    is_lifecycle: bool = False  # connectedCallback, renderedCallback, etc.
    docstring: Optional[str] = None


@dataclass
class LWCComponent:
    """Represents a complete LWC component."""

    name: str
    class_name: str
    extends: str  # Usually LightningElement
    api_properties: List[LWCProperty] = field(default_factory=list)
    tracked_properties: List[LWCProperty] = field(default_factory=list)
    wire_adapters: List[LWCProperty] = field(default_factory=list)
    methods: List[LWCMethod] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    apex_imports: List[str] = field(default_factory=list)
    template_bindings: List[str] = field(default_factory=list)
    custom_events: List[str] = field(default_factory=list)
    package: str = ""
    js_file: str = ""
    html_file: str = ""


class LWCProcessor(IProcessor):
    """
    Process Lightning Web Component files.

    Extracts:
    - Component class and properties
    - @api, @track, @wire decorators
    - Apex method imports
    - Template data bindings
    - Event handlers and custom events

    Example:
        processor = LWCProcessor()
        result = processor.process(Path("myComponent/myComponent.js"))
        print(result.metadata["api_properties"])
        print(result.metadata["apex_imports"])
    """

    # Decorator patterns
    API_DECORATOR = re.compile(r"@api\s+(?:get\s+)?(\w+)")
    TRACK_DECORATOR = re.compile(r"@track\s+(\w+)")
    WIRE_DECORATOR = re.compile(
        r"@wire\s*\(\s*(\w+)(?:\s*,\s*\{([^}]*)\})?\s*\)\s*(\w+)", re.DOTALL
    )

    # Import patterns
    IMPORT_PATTERN = re.compile(
        r'import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]',
    )
    APEX_IMPORT = re.compile(r"@salesforce/apex/(\w+)\.(\w+)")

    # Class pattern
    CLASS_PATTERN = re.compile(
        r"export\s+default\s+class\s+(\w+)\s+extends\s+(\w+)",
    )

    # Method patterns
    METHOD_PATTERN = re.compile(
        r"(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{",
    )
    GETTER_PATTERN = re.compile(r"get\s+(\w+)\s*\(\)")
    SETTER_PATTERN = re.compile(r"set\s+(\w+)\s*\(")

    # Template patterns (for HTML)
    BINDING_PATTERN = re.compile(r"\{(\w+(?:\.\w+)*)\}")
    FOR_EACH_PATTERN = re.compile(r'for:each="\{(\w+)\}"')
    IF_PATTERN = re.compile(r'if:(?:true|false)="\{(\w+)\}"')
    ITERATOR_PATTERN = re.compile(r'iterator:(\w+)="\{(\w+)\}"')

    # Event patterns
    DISPATCH_EVENT = re.compile(
        r'this\.dispatchEvent\s*\(\s*new\s+CustomEvent\s*\(\s*[\'"](\w+)[\'"]'
    )
    HANDLER_PREFIX = re.compile(r"^handle[A-Z]")

    # Lifecycle methods
    LIFECYCLE_METHODS = {
        "constructor",
        "connectedCallback",
        "disconnectedCallback",
        "renderedCallback",
        "errorCallback",
        "render",
    }

    def __init__(self, sfdx_parser: Any = None) -> None:
        """
        Initialize LWC processor.

        Args:
            sfdx_parser: Optional SFDXProjectParser for package resolution
        """
        self.sfdx_parser = sfdx_parser

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        # Only process the main component JS file (not tests, not __tests__)
        if file_path.suffix.lower() != ".js":
            return False
        if "__tests__" in str(file_path):
            return False
        # Check if it's in an lwc directory structure
        return "lwc" in str(file_path).lower()

    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".js"]

    def process(self, file_path: Path) -> ExtractedContent:
        """
        Process an LWC component.

        Args:
            file_path: Path to the component's .js file

        Returns:
            ExtractedContent with parsed LWC data
        """
        js_content = self._read_file(file_path)
        component = self._parse_component(js_content, file_path)

        # Try to find and parse the HTML template
        html_file = file_path.with_suffix(".html")
        if html_file.exists():
            html_content = self._read_file(html_file)
            self._parse_template(html_content, component)
            component.html_file = str(html_file)

        # Resolve package
        component.package = self._resolve_package(file_path)

        # Build text content for indexing
        text_content = self._build_text_content(component, js_content)

        # Build source location
        source_location = SourceLocation(
            source_type=SourceType.CODE,
            title=component.name,
            file_path=str(file_path),
        )

        return ExtractedContent(
            text=text_content,
            metadata={
                "name": component.name,
                "class_name": component.class_name,
                "extends": component.extends,
                "package": component.package,
                "api_properties": [
                    self._property_to_dict(p) for p in component.api_properties
                ],
                "tracked_properties": [
                    self._property_to_dict(p) for p in component.tracked_properties
                ],
                "wire_adapters": [
                    self._property_to_dict(p) for p in component.wire_adapters
                ],
                "methods": [self._method_to_dict(m) for m in component.methods],
                "imports": component.imports,
                "apex_imports": component.apex_imports,
                "template_bindings": component.template_bindings,
                "custom_events": component.custom_events,
                "js_file": str(file_path),
                "html_file": component.html_file,
                "api_count": len(component.api_properties),
                "wire_count": len(component.wire_adapters),
                "method_count": len(component.methods),
            },
            sections=[m.name for m in component.methods],
        )

    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return file_path.read_bytes().decode("utf-8", errors="ignore")

    def _parse_component(self, content: str, file_path: Path) -> LWCComponent:
        """Parse the LWC JavaScript file."""
        # Get component name from directory
        component_name = file_path.parent.name

        # Parse class declaration
        class_match = self.CLASS_PATTERN.search(content)
        if class_match:
            class_name = class_match.group(1)
            extends = class_match.group(2)
        else:
            class_name = component_name
            extends = "LightningElement"

        component = LWCComponent(
            name=component_name,
            class_name=class_name,
            extends=extends,
            js_file=str(file_path),
        )

        # Parse imports
        self._parse_imports(content, component)

        # Parse decorators
        self._parse_decorators(content, component)

        # Parse methods
        self._parse_methods(content, component)

        # Find custom events
        for match in self.DISPATCH_EVENT.finditer(content):
            event_name = match.group(1)
            if event_name not in component.custom_events:
                component.custom_events.append(event_name)

        return component

    def _parse_imports(self, content: str, component: LWCComponent) -> None:
        """Parse import statements.

        Rule #1: Reduced nesting via helper extraction
        """
        for match in self.IMPORT_PATTERN.finditer(content):
            imports = match.group(1) or match.group(2)
            source = match.group(3)

            # Record all imports
            self._add_imports_from_match(imports, source, component)

            # Check for Apex imports
            self._add_apex_import_if_present(source, component)

    def _add_imports_from_match(
        self, imports: str, source: str, component: LWCComponent
    ) -> None:
        """Add imports from regex match to component.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if not imports:
            return

        for imp in imports.split(","):
            imp_name = imp.strip()
            if imp_name:
                component.imports.append(f"{imp_name} from {source}")

    def _add_apex_import_if_present(self, source: str, component: LWCComponent) -> None:
        """Add Apex import if source is an Apex method.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        apex_match = self.APEX_IMPORT.search(source)
        if apex_match:
            controller = apex_match.group(1)
            method = apex_match.group(2)
            component.apex_imports.append(f"{controller}.{method}")

    def _parse_decorators(self, content: str, component: LWCComponent) -> None:
        """Parse @api, @track, and @wire decorators.

        Rule #1: Reduced nesting via helper extraction
        """
        # Parse @api properties
        for match in self.API_DECORATOR.finditer(content):
            prop_name = match.group(1)
            prop = LWCProperty(
                name=prop_name,
                decorator="@api",
                is_getter="get " in content[max(0, match.start() - 10) : match.start()],
            )
            component.api_properties.append(prop)

        # Parse @track properties
        for match in self.TRACK_DECORATOR.finditer(content):
            prop_name = match.group(1)
            prop = LWCProperty(
                name=prop_name,
                decorator="@track",
            )
            component.tracked_properties.append(prop)

        # Parse @wire decorators
        for match in self.WIRE_DECORATOR.finditer(content):
            adapter = match.group(1)
            config = match.group(2)
            prop_name = match.group(3)

            params = self._parse_wire_config(config)

            prop = LWCProperty(
                name=prop_name,
                decorator="@wire",
                wire_adapter=adapter,
                wire_params=params if params else None,
            )
            component.wire_adapters.append(prop)

    def _parse_wire_config(self, config: str) -> dict[str, Any]:
        """Parse @wire decorator configuration parameters.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        params = {}
        if not config:
            return params

        for param in config.split(","):
            param = param.strip()
            if ":" in param:
                key, value = param.split(":", 1)
                params[key.strip()] = value.strip()

        return params

    def _parse_methods(self, content: str, component: LWCComponent) -> None:
        """Parse class methods."""
        # Find the class body
        class_match = self.CLASS_PATTERN.search(content)
        if not class_match:
            return

        class_start = class_match.end()

        # Find methods
        seen_methods: Set[str] = set()

        for match in self.METHOD_PATTERN.finditer(content[class_start:]):
            method_name = match.group(1)

            # Skip if already seen or is a keyword
            if method_name in seen_methods:
                continue
            if method_name in [
                "if",
                "for",
                "while",
                "switch",
                "catch",
                "return",
                "async",
            ]:
                continue

            seen_methods.add(method_name)

            params = match.group(2)
            parameters = [p.strip() for p in params.split(",") if p.strip()]

            method = LWCMethod(
                name=method_name,
                parameters=parameters,
                is_handler=bool(self.HANDLER_PREFIX.match(method_name)),
                is_lifecycle=method_name in self.LIFECYCLE_METHODS,
            )
            component.methods.append(method)

        # Also find getters
        for match in self.GETTER_PATTERN.finditer(content[class_start:]):
            getter_name = match.group(1)
            if getter_name not in seen_methods:
                seen_methods.add(getter_name)
                method = LWCMethod(
                    name=getter_name,
                    is_handler=False,
                    is_lifecycle=False,
                )
                component.methods.append(method)

    def _parse_template(self, html_content: str, component: LWCComponent) -> None:
        """Parse HTML template for bindings."""
        bindings: Set[str] = set()

        # Find all data bindings
        for match in self.BINDING_PATTERN.finditer(html_content):
            binding = match.group(1)
            # Get root property
            root = binding.split(".")[0]
            bindings.add(root)

        # Find for:each bindings
        for match in self.FOR_EACH_PATTERN.finditer(html_content):
            bindings.add(match.group(1))

        # Find if bindings
        for match in self.IF_PATTERN.finditer(html_content):
            bindings.add(match.group(1))

        # Find iterator bindings
        for match in self.ITERATOR_PATTERN.finditer(html_content):
            bindings.add(match.group(2))

        component.template_bindings = sorted(bindings)

    def _resolve_package(self, file_path: Path) -> str:
        """Resolve package name from file path."""
        if self.sfdx_parser:
            pkg = self.sfdx_parser.get_package_for_file(file_path)
            if pkg:
                return pkg

        # Fallback: parse from path
        path_str = str(file_path).replace("\\", "/")
        match = re.search(r"/src/([a-zA-Z0-9_-]+)/", path_str)
        if match:
            return match.group(1)

        return ""

    def _build_text_content(self, component: LWCComponent, js_content: str) -> str:
        """Build searchable text content.

        Rule #1: Reduced nesting via section extraction
        """
        parts = []

        parts.append(f"LWC Component: {component.name}")
        parts.append(f"Class: {component.class_name} extends {component.extends}")

        if component.package:
            parts.append(f"Package: {component.package}")

        # Add component sections
        self._add_properties_section(component, parts)
        self._add_wire_adapters_section(component, parts)
        self._add_apex_imports_section(component, parts)
        self._add_methods_section(component, parts)
        self._add_events_section(component, parts)
        self._add_template_bindings_section(component, parts)

        # Include full source
        parts.append("\n\n--- Source Code ---\n")
        parts.append(js_content)

        return "\n".join(parts)

    def _add_properties_section(
        self, component: LWCComponent, parts: list[Any]
    ) -> None:
        """Add @api and @track properties to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # API properties
        if component.api_properties:
            parts.append("\n@api Properties:")
            for prop in component.api_properties:
                parts.append(f"  - {prop.name}")

        # Tracked properties
        if component.tracked_properties:
            parts.append("\n@track Properties:")
            for prop in component.tracked_properties:
                parts.append(f"  - {prop.name}")

    def _add_wire_adapters_section(
        self, component: LWCComponent, parts: list[Any]
    ) -> None:
        """Add @wire adapters to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if component.wire_adapters:
            parts.append("\n@wire Adapters:")
            for prop in component.wire_adapters:
                parts.append(f"  - {prop.name}: {prop.wire_adapter}")

    def _add_apex_imports_section(
        self, component: LWCComponent, parts: list[Any]
    ) -> None:
        """Add Apex imports to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if component.apex_imports:
            parts.append("\nApex Imports:")
            for apex in component.apex_imports:
                parts.append(f"  - {apex}")

    def _add_methods_section(self, component: LWCComponent, parts: list[Any]) -> None:
        """Add methods to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if not component.methods:
            return

        parts.append("\nMethods:")
        for method in component.methods:
            method_type = self._get_method_type_label(method)
            params = ", ".join(method.parameters)
            parts.append(f"  - {method.name}({params}){method_type}")

    def _get_method_type_label(self, method: Any) -> str:
        """Get method type label.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if method.is_lifecycle:
            return " (lifecycle)"
        if method.is_handler:
            return " (handler)"
        return ""

    def _add_events_section(self, component: LWCComponent, parts: list[Any]) -> None:
        """Add custom events to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if component.custom_events:
            parts.append("\nCustom Events:")
            for event in component.custom_events:
                parts.append(f"  - {event}")

    def _add_template_bindings_section(
        self, component: LWCComponent, parts: list[Any]
    ) -> None:
        """Add template bindings to parts.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if component.template_bindings:
            parts.append("\nTemplate Bindings:")
            parts.append(f"  {', '.join(component.template_bindings)}")

    def _property_to_dict(self, prop: LWCProperty) -> Dict[str, Any]:
        """Convert property to dictionary."""
        result = {
            "name": prop.name,
            "decorator": prop.decorator,
        }
        if prop.wire_adapter:
            result["wire_adapter"] = prop.wire_adapter
        if prop.wire_params:
            result["wire_params"] = prop.wire_params
        if prop.default_value:
            result["default_value"] = prop.default_value
        if prop.is_getter:
            result["is_getter"] = True
        return result

    def _method_to_dict(self, method: LWCMethod) -> Dict[str, Any]:
        """Convert method to dictionary."""
        return {
            "name": method.name,
            "parameters": method.parameters,
            "is_handler": method.is_handler,
            "is_lifecycle": method.is_lifecycle,
        }
