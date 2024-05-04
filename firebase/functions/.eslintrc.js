/** @format */

module.exports = {
	root: true,
	env: {
		es6: true,
		node: true,
	},
	extends: [
		'eslint:recommended',
		'plugin:import/errors',
		'plugin:import/warnings',
		'plugin:import/typescript',
		'google',
		'plugin:@typescript-eslint/recommended',
	],
	parser: '@typescript-eslint/parser',
	parserOptions: {
		project: ['tsconfig.json', 'tsconfig.dev.json'],
		sourceType: 'module',
	},
	ignorePatterns: [
		'/lib/**/*', // Ignore built files.
		'/generated/**/*', // Ignore generated files.
	],
	plugins: ['@typescript-eslint', 'import'],
	rules: {
		quotes: ['error', 'double'],
		'import/no-unresolved': 0,
		indent: ['error', 2],
		'object-curly-spacing': 'off',
		indent: 'off',
		'no-tabs': 'off',
		quotes: 'off',
		'quote-props': 'off',
		'no-dupe-keys': 'off',
		'@typescript-eslint/no-var-requires': 'off',
		'@typescript-eslint/no-explicit-any': 'off',
	},
};
