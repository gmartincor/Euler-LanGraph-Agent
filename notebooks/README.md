# Notebooks para Desarrollo

Este directorio contiene notebooks de Jupyter para desarrollo y testing de componentes del ReAct Agent.

## Estructura

- `01_environment_test.ipynb` - Verificar dependencias y configuración
- `02_math_tools_test.ipynb` - Probar cálculo de integrales y visualización  
- `03_agent_test.ipynb` - Probar agente LangGraph con herramientas
- `04_database_test.ipynb` - Probar conexión y persistencia PostgreSQL

## Uso

```bash
# Instalar Jupyter en el entorno
poetry add jupyter

# Ejecutar Jupyter
poetry run jupyter lab notebooks/

# O usar VS Code con extensión Jupyter
```

## Principios

- **Modularidad**: Cada notebook prueba un componente específico
- **Iterativo**: Desarrollo incremental y validación paso a paso  
- **Reutilizable**: Funciones que se pueden extraer al código principal
- **Documentado**: Explicaciones claras en markdown

## Flujo de Desarrollo

1. Probar componentes individualmente en notebooks
2. Validar integración entre componentes
3. Extraer código estable a módulos Python
4. Mantener notebooks como documentación viva
