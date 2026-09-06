# Target configuration

One immutable YAML document describes each target used by a campaign. Copy `_template.yaml`, fill every required field, validate it, and record its SHA-256 in the campaign proposal and run manifest.

Target configuration contains scientific metadata and stable data-product identifiers. Site-specific mount paths belong in deployment configuration. Priors are explicit distributions with units and provenance; catalogue values are not silently converted into fixed parameters.

Production code must receive the resolved configuration through a loader or CLI argument. It must not select behavior from a target name.
