rule.conditions.map((condition, index) => (
                        <div
                          key={condition.id}
                          className="flex items-center space-x-3 p-3 bg-surface-800 rounded-lg"
                        >
                          <GripVertical className="w-4 h-4 text-surface-600 cursor-grab" />
                          
                          {/* Indicator */}
                          <select
                            value={condition.indicator}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, indicator: e.target.value }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm min-w-[120px]"
                          >
                            {indicators.map((group) => (
                              <optgroup key={group.group} label={group.group}>
                                {group.items.map((item) => (
                                  <option key={item} value={item}>
                                    {item}
                                  </option>
                                ))}
                              </optgroup>
                            ))}
                          </select>

                          {/* Operator */}
                          <select
                            value={condition.operator}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, operator: e.target.value as ConditionOperator }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm"
                          >
                            {operators.map((op) => (
                              <option key={op.value} value={op.value}>
                                {op.label}
                              </option>
                            ))}
                          </select>

                          {/* Value */}
                          <input
                            type="text"
                            value={condition.value}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, value: e.target.value }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm w-24"
                            placeholder="Değer"
                          />

                          {/* Delete */}
                          <button
                            onClick={() => removeCondition(rule.id, condition.id)}
                            className="p-1.5 text-surface-500 hover:text-red-400 transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>

                          {/* Logic connector */}
                          {index < rule.conditions.length - 1 && (
                            <span className="text-xs text-primary-400 font-medium px-2">
                              {rule.logic}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Add Condition Button */}
                    <div className="relative">
                      <button
                        onClick={() => setShowIndicatorPicker(showIndicatorPicker === rule.id ? false : rule.id as any)}
                        className="flex items-center space-x-2 px-4 py-2 border border-dashed border-surface-600 rounded-lg text-surface-400 hover:text-white hover:border-surface-500 transition-colors w-full justify-center"
                      >
                        <Plus className="w-4 h-4" />
                        <span>Koşul Ekle</span>
                      </button>

                      {/* Indicator Picker Dropdown */}
                      {showIndicatorPicker === rule.id && (
                        <div className="absolute top-full left-0 right-0 mt-2 bg-surface-800 border border-surface-700 rounded-lg shadow-xl z-10 max-h-64 overflow-y-auto">
                          {indicators.map((group) => (
                            <div key={group.group}>
                              <div className="px-3 py-2 text-xs text-surface-500 font-medium bg-surface-900">
                                {group.group}
                              </div>
                              {group.items.map((item) => (
                                <button
                                  key={item}
                                  onClick={() => addCondition(rule.id, item)}
                                  className="w-full px-4 py-2 text-left text-sm hover:bg-surface-700 transition-colors"
                                >
                                  {item}
                                </button>
                              ))}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Empty State */}
          {strategy.rules.length === 0 && (
            <div className="bg-surface-900 rounded-xl border border-surface-800 p-12 text-center">
              <Zap className="w-12 h-12 text-surface-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">Strateji Boş</h3>
              <p className="text-surface-400 mb-4">
                Sol panelden kural ekleyerek başlayın
              </p>
            </div>
          )}

          {/* Strategy Preview */}
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-4">
            <h2 className="font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2 text-yellow-500" />
              Strateji Özeti
            </h2>
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-xs text-surface-500 mb-1">Entry Kuralları</p>
                <p className="text-xl font-bold text-green-400">
                  {strategy.rules.filter((r) => r.type === 'entry').length}
                </p>
              </div>
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-xs text-surface-500 mb-1">Exit Kuralları</p>
                <p className="text-xl font-bold text-red-400">
                  {strategy.rules.filter((r) => r.type === 'exit').length}
                </p>
              </div>
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-xs text-surface-500 mb-1">Filtreler</p>
                <p className="text-xl font-bold text-blue-400">
                  {strategy.rules.filter((r) => r.type === 'filter').length}
                </p>
              </div>
              <div className="bg-surface-800 rounded-lg p-3">
                <p className="text-xs text-surface-500 mb-1">Toplam Koşul</p>
                <p className="text-xl font-bold text-white">
                  {strategy.rules.reduce((acc, r) => acc + r.conditions.length, 0)}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
rule.conditions.map((condition, idx) => (
                        <div
                          key={condition.id}
                          className="flex items-center space-x-3 p-3 bg-surface-800 rounded-lg"
                        >
                          <span className="text-xs text-surface-500 w-6">{idx + 1}.</span>
                          
                          {/* Indicator */}
                          <select
                            value={condition.indicator}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, indicator: e.target.value }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm flex-1"
                          >
                            {indicators.map((group) => (
                              <optgroup key={group.group} label={group.group}>
                                {group.items.map((item) => (
                                  <option key={item} value={item}>
                                    {item}
                                  </option>
                                ))}
                              </optgroup>
                            ))}
                          </select>

                          {/* Operator */}
                          <select
                            value={condition.operator}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, operator: e.target.value as ConditionOperator }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm w-40"
                          >
                            {operators.map((op) => (
                              <option key={op.value} value={op.value}>
                                {op.label}
                              </option>
                            ))}
                          </select>

                          {/* Value */}
                          <input
                            type="text"
                            value={condition.value}
                            onChange={(e) =>
                              setStrategy((prev) => ({
                                ...prev,
                                rules: prev.rules.map((r) =>
                                  r.id === rule.id
                                    ? {
                                        ...r,
                                        conditions: r.conditions.map((c) =>
                                          c.id === condition.id
                                            ? { ...c, value: e.target.value }
                                            : c
                                        ),
                                      }
                                    : r
                                ),
                              }))
                            }
                            className="px-3 py-1.5 bg-surface-700 border border-surface-600 rounded-lg text-sm w-32"
                            placeholder="Değer"
                          />

                          {/* Remove */}
                          <button
                            onClick={() => removeCondition(rule.id, condition.id)}
                            className="p-1.5 text-surface-500 hover:text-red-400 transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>

                    {/* Add Condition */}
                    <button
                      onClick={() => {
                        setShowIndicatorPicker(true);
                        setEditingCondition(rule.id);
                      }}
                      className="flex items-center space-x-2 px-4 py-2 border border-dashed border-surface-600 rounded-lg text-surface-400 hover:border-surface-500 hover:text-white transition-colors w-full justify-center"
                    >
                      <Plus className="w-4 h-4" />
                      <span>Koşul Ekle</span>
                    </button>
                  </div>
                )}
              </div>
            );
          })}

          {strategy.rules.length === 0 && (
            <div className="bg-surface-900 rounded-xl border border-surface-800 p-12 text-center">
              <Zap className="w-12 h-12 text-surface-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">Strateji Boş</h3>
              <p className="text-surface-400 mb-4">
                Sol panelden kural ekleyerek stratejinizi oluşturmaya başlayın
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Indicator Picker Modal */}
      {showIndicatorPicker && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-surface-900 rounded-xl border border-surface-800 w-[500px] max-h-[600px] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-surface-800">
              <h3 className="font-semibold text-white">Gösterge Seç</h3>
              <button
                onClick={() => setShowIndicatorPicker(false)}
                className="p-1 text-surface-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto max-h-[500px]">
              {indicators.map((group) => (
                <div key={group.group} className="mb-4">
                  <h4 className="text-sm font-medium text-surface-400 mb-2">{group.group}</h4>
                  <div className="grid grid-cols-3 gap-2">
                    {group.items.map((item) => (
                      <button
                        key={item}
                        onClick={() => {
                          if (editingCondition) {
                            addCondition(editingCondition, item);
                          }
                        }}
                        className="px-3 py-2 bg-surface-800 rounded-lg text-sm text-white hover:bg-surface-700 transition-colors text-left"
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
